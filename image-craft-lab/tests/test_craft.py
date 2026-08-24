"""
Tests for image-craft-lab — 3 studios.
Run: pytest -q
Generated images only, no Streamlit.
"""

import re
import sys

import numpy as np
import pytest
from PIL import Image

from src.ascii import image_to_ascii
from src.palette import contrast_ratio, extract_palette
from src.pixel import image_to_pixel
from src.presets import (
    CHAR_ASPECT,
    CHARSET_BW,
    CHARSET_CLASSIC,
    CHARSET_DETAILED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def solid(color, size=(100, 60)):
    return Image.new("RGB", size, color)


def gradient_image(w=120, h=80):
    """Horizontal black→white gradient."""
    img = Image.new("RGB", (w, h))
    for x in range(w):
        v = int(255 * x / (w - 1)) if w > 1 else 0
        for y in range(h):
            img.putpixel((x, y), (v, v, v))
    return img


def colorful_image():
    """2×2 distinct colors."""
    img = Image.new("RGB", (4, 4))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for y in range(4):
        for x in range(4):
            img.putpixel((x, y), colors[(x + y) % len(colors)])
    return img


# ---------------------------------------------------------------------------
# ASCII
# ---------------------------------------------------------------------------
def test_ascii_black_maps_to_first_char():
    # black luminance 0 → charset[0]
    img = solid((0, 0, 0), size=(80, 40))
    out = image_to_ascii(img, cols=40, charset=CHARSET_CLASSIC, color_mode="grayscale", contrast=1.0, brightness=1.0, invert=False)
    # All chars should be CHARSET_CLASSIC[0] == "@"
    flat = out["text"].replace("\n", "")
    assert all(c == CHARSET_CLASSIC[0] for c in flat), f"expected all '{CHARSET_CLASSIC[0]}', got {set(flat)}"


def test_ascii_white_maps_to_last_char():
    img = solid((255, 255, 255), size=(80, 40))
    out = image_to_ascii(img, cols=40, charset=CHARSET_CLASSIC, color_mode="grayscale")
    flat = out["text"].replace("\n", "")
    assert all(c == CHARSET_CLASSIC[-1] for c in flat), f"expected all '{CHARSET_CLASSIC[-1]}', got {set(flat)}"


def test_ascii_cols_clamp():
    img = solid((128, 128, 128))
    # below min
    out_low = image_to_ascii(img, cols=5, charset=CHARSET_CLASSIC)
    assert out_low["width"] == 40  # clamped to min
    # above max
    out_high = image_to_ascii(img, cols=999, charset=CHARSET_CLASSIC)
    assert out_high["width"] == 300  # clamped to max
    # height aspect
    # rows = int(cols * (h/w) * 0.55)
    w, h = img.size
    expected_rows_low = int(40 * (h / w) * CHAR_ASPECT)
    assert out_low["height"] == max(1, expected_rows_low)


def test_ascii_bw_two_char_only():
    img = gradient_image(100, 40)
    out = image_to_ascii(img, cols=20, charset=CHARSET_CLASSIC, color_mode="bw")
    flat = out["text"].replace("\n", "")
    # bw uses first and last char of charset (threshold 128)
    c_dark = CHARSET_CLASSIC[0]
    c_light = CHARSET_CLASSIC[-1]
    assert set(flat).issubset({c_dark, c_light}), f"bw should be 2 chars {c_dark}/{c_light}, got {set(flat)}"
    assert len(set(flat)) == 2  # gradient should hit both sides


def test_ascii_invert_swaps():
    img = solid((0, 0, 0), size=(60, 30))
    normal = image_to_ascii(img, cols=40, charset=CHARSET_CLASSIC, invert=False)
    inverted = image_to_ascii(img, cols=40, charset=CHARSET_CLASSIC, invert=True)
    # Normal black → first char, inverted black→ white → last char
    assert normal["text"][0] == CHARSET_CLASSIC[0]
    assert inverted["text"][0] == CHARSET_CLASSIC[-1]
    # And white swapped
    img_w = solid((255, 255, 255), size=(60, 30))
    assert image_to_ascii(img_w, cols=40, charset=CHARSET_CLASSIC, invert=False)["text"][0] == CHARSET_CLASSIC[-1]
    assert image_to_ascii(img_w, cols=40, charset=CHARSET_CLASSIC, invert=True)["text"][0] == CHARSET_CLASSIC[0]


def test_ascii_contrast_brightness_do_not_crash_and_html():
    img = solid((100, 150, 200), size=(60, 40))
    out = image_to_ascii(img, cols=50, charset=CHARSET_DETAILED, color_mode="original", contrast=1.5, brightness=0.8)
    assert out["width"] == 50
    assert out["height"] >= 1
    assert "html" in out and "<span" in out["html"]
    # grayscale html has <pre>
    out2 = image_to_ascii(img, cols=50, charset=CHARSET_CLASSIC, color_mode="grayscale")
    assert "<pre" in out2["html"]


# ---------------------------------------------------------------------------
# PIXEL
# ---------------------------------------------------------------------------
def test_pixel_output_size_grid_scaled():
    img = solid((120, 80, 200), size=(200, 100))
    grid, scale = 64, 8
    out = image_to_pixel(img, grid=grid, colors=16, dither=False, scale=scale)
    # small_h = int(grid * h/w) => int(64 * 100/200)=32
    assert out["small_w"] == grid
    assert out["small_h"] == int(grid * 100 / 200)
    assert out["width"] == out["small_w"] * scale
    assert out["height"] == out["small_h"] * scale
    assert out["image"].size == (out["width"], out["height"])


def test_pixel_colors_lte_requested():
    # Use colorful image with many colors
    img = gradient_image(100, 100)
    # Convert gradient to RGB already
    out = image_to_pixel(img, grid=32, colors=8, dither=False, scale=4)
    # Count unique colors in small quantized image
    small = out["small"]
    uniq = set(small.getdata())
    assert len(uniq) <= 8, f"expected <=8 colors, got {len(uniq)}"
    # Also test 4 colors
    out2 = image_to_pixel(img, grid=32, colors=4, dither=False, scale=4)
    uniq2 = set(out2["small"].getdata())
    assert len(uniq2) <= 4


def test_pixel_dither_flag_doesnt_crash():
    img = solid((200, 100, 50), size=(80, 80))
    for d in [False, True]:
        out = image_to_pixel(img, grid=16, colors=16, dither=d, scale=8)
        assert out["image"] is not None
        assert out["css"] is not None
        # css should contain px and hex
        assert "px" in out["css"]
        assert "#" in out["css"]


def test_pixel_css_generation():
    img = Image.new("RGB", (2, 2))
    img.putpixel((0, 0), (255, 0, 0))
    img.putpixel((1, 0), (0, 255, 0))
    img.putpixel((0, 1), (0, 0, 255))
    img.putpixel((1, 1), (255, 255, 255))
    out = image_to_pixel(img, grid=2, colors=4, dither=False, scale=10)
    css = out["css"]
    # Should have 2*2=4 shadows (or up to grid scaled)
    # At grid=2, small is 2x2, so 4 entries
    # Each entry has "px"
    assert css.count("px") >= 4


# ---------------------------------------------------------------------------
# PALETTE
# ---------------------------------------------------------------------------
def test_palette_len_hex_equals_n_colors():
    img = solid((120, 180, 200), size=(100, 100))
    for n in [3, 5, 8]:
        pal = extract_palette(img, n_colors=n, mode="vibrant")
        assert len(pal["hex"]) == n, f"n={n} expected {n} hex, got {len(pal['hex'])}"
        assert len(pal["rgb"]) == n
        assert len(pal["counts"]) == n


def test_palette_hex_format():
    img = colorful_image()
    pal = extract_palette(img, n_colors=4, mode="vibrant")
    pat = re.compile(r"^#[0-9a-f]{6}$")
    for h in pal["hex"]:
        assert pat.match(h), f"hex format bad: {h}"


def test_palette_vibrant_vs_pastel_differ():
    # Use saturated image
    img = solid((220, 20, 60), size=(100, 100))
    vib = extract_palette(img, n_colors=4, mode="vibrant")
    past = extract_palette(img, n_colors=4, mode="pastel")
    # Pastel should be different (lighter / less saturated)
    assert vib["hex"] != past["hex"], "vibrant vs pastel should differ"
    # Pastel hex should be lighter on average? Check value
    # Convert to rgb and compare avg luminance
    def avg_lum(hexes):
        return sum(sum(int(h[i:i+2],16) for i in (1,3,5))/3 for h in hexes)/len(hexes)
    # Pastel lighter?
    # Not strict, but at least one channel differs
    diff = sum(1 for a,b in zip(vib["hex"], past["hex"]) if a!=b)
    assert diff >= 1


def test_palette_counts_sum():
    img = gradient_image(100, 100)
    pal = extract_palette(img, n_colors=5, mode="muted")
    # counts are pixel counts in 100x100 cluster assignments
    # For KMeans, sum should be 10000; for MEDIANCUT, sum is also 10000 (100*100)
    total = sum(pal["counts"])
    assert total == 10000, f"counts sum should be 10000 (100×100), got {total}"


def test_palette_missing_sklearn_fallback():
    # Force sklearn import failure by patching sys.modules
    # Note: palette already caches sklearn availability per call via try/except inside _extract_via_kmeans
    # So we simulate by temporarily hiding sklearn
    import unittest.mock as mock

    img = solid((80, 120, 200), size=(100, 100))
    # Patch import to fail
    with mock.patch.dict("sys.modules", {"sklearn": None, "sklearn.cluster": None}):
        # Also patch the helper to return None to force fallback path
        # Instead we directly test that palette still returns n_colors even when KMeans unavailable
        # Call via mediancut path by monkeypatching the kmeans helper
        import src.palette as pal_mod
        orig = pal_mod._extract_via_kmeans
        pal_mod._extract_via_kmeans = lambda pixels, n: (None, None)
        try:
            pal = extract_palette(img, n_colors=6, mode="deep")
            assert len(pal["hex"]) == 6
            assert len(pal["rgb"]) == 6
        finally:
            pal_mod._extract_via_kmeans = orig


def test_palette_wcag_contrast():
    # Known values
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, rel=1e-2)
    assert contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0, rel=1e-2)
    assert contrast_ratio("#ff0000", "#ff0000") == pytest.approx(1.0, rel=1e-2)
    # Palette wcag dict exists
    img = solid((0, 0, 0), size=(50, 50))
    pal = extract_palette(img, n_colors=3, mode="vibrant")
    assert "wcag" in pal
    assert "white_ratio" in pal["wcag"]
    assert "black_ratio" in pal["wcag"]
    assert pal["wcag"]["dominant"] in pal["hex"]


def test_palette_gradient_size_and_css_vars():
    img = solid((100, 150, 200), size=(60, 60))
    pal = extract_palette(img, n_colors=5, mode="vibrant")
    grad = pal["gradient"]
    assert grad.size == (800, 120)
    assert grad.mode == "RGB"
    assert ":root" in pal["css_vars"]
    assert "--color-1" in pal["css_vars"]
    assert "colors:" in pal["css_vars"]


def test_palette_mode_transforms_clamped():
    # Ensure muted/pastel/deep don't produce out-of-range rgb
    for mode in ["vibrant", "muted", "pastel", "deep"]:
        img = colorful_image()
        pal = extract_palette(img, n_colors=5, mode=mode)
        for r, g, b in pal["rgb"]:
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255
