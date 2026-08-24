"""
Palette engine — pure functions, no Streamlit imports.

Algorithm (exact):
  1. Resize to 100×100 (BILINEAR), reshape(-1,3).
  2. If sklearn available: KMeans(n_clusters=n_colors, n_init=5, random_state=0) → centers + counts.
     Else: image.quantize(colors=n_colors, method=MEDIANCUT) → palette.
  3. Sort by counts descending (dominant first).
  4. mode transforms: muted → saturation*0.6, pastel → saturation*0.5 + lightness+0.25,
     deep → value*0.7. Via colorsys.rgb_to_hsv → adjust → hsv_to_rgb. Clamp 0–255.
  5. Generate css_vars: :root { --color-1: #hex; } + Tailwind colors snippet.
  6. Generate gradient as 800×120 linear gradient of the hexes.
  7. WCAG contrast: compute contrast_ratio(hex1, hex2) for dominant vs white/black.
"""

from __future__ import annotations

import colorsys
from typing import List, Tuple

import numpy as np
from PIL import Image

N_MIN, N_MAX = 3, 8


def _clamp_n(n: int) -> int:
    return max(N_MIN, min(N_MAX, int(n)))


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _transform_color(r: int, g: int, b: int, mode: str) -> Tuple[int, int, int]:
    """Apply mode transforms via HSV."""
    # Normalize to 0-1
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rn, gn, bn)
    # Use HSV as H, S, V (note colorsys uses V as value/brightness, not L)
    # For muted/pastel/deep transforms as spec:
    if mode == "muted":
        s = s * 0.6
    elif mode == "pastel":
        s = s * 0.5
        # lightness+0.25 — hsv v approximates value; increase v
        v = min(1.0, v + 0.25)
        # also slightly boost if too dark
        # Alternative pastel via mix with white could be done but spec says s*0.5 + lightness+0.25
    elif mode == "deep":
        v = v * 0.7
    # vibrant = no change
    rn2, gn2, bn2 = colorsys.hsv_to_rgb(h, s, v)
    return (
        int(round(max(0.0, min(1.0, rn2)) * 255)),
        int(round(max(0.0, min(1.0, gn2)) * 255)),
        int(round(max(0.0, min(1.0, bn2)) * 255)),
    )


def _luminance_channel(c: float) -> float:
    """Helper for WCAG relative luminance."""
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(r: int, g: int, b: int) -> float:
    """WCAG relative luminance."""
    return 0.2126 * _luminance_channel(r) + 0.7152 * _luminance_channel(g) + 0.0722 * _luminance_channel(b)


def contrast_ratio(hex1: str, hex2: str) -> float:
    """WCAG contrast ratio between two hex colors."""
    r1, g1, b1 = _hex_to_rgb(hex1)
    r2, g2, b2 = _hex_to_rgb(hex2)
    l1 = relative_luminance(r1, g1, b1)
    l2 = relative_luminance(r2, g2, b2)
    # lighter / darker
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _css_vars(hexes: List[str]) -> str:
    lines = [":root {"]
    for i, h in enumerate(hexes, 1):
        lines.append(f"  --color-{i}: {h};")
    lines.append("}")
    lines.append("")
    lines.append("/* Tailwind */")
    lines.append("colors: {")
    for i, h in enumerate(hexes, 1):
        lines.append(f'  "color-{i}": "{h}",')
    lines.append("}")
    return "\n".join(lines)


def _generate_gradient(hexes: List[str], width: int = 800, height: int = 120) -> Image.Image:
    """Create linear gradient image interpolating between hexes."""
    if not hexes:
        return Image.new("RGB", (width, height), (0, 0, 0))
    if len(hexes) == 1:
        r, g, b = _hex_to_rgb(hexes[0])
        return Image.new("RGB", (width, height), (r, g, b))

    # Convert hexes to rgb array
    rgbs = [_hex_to_rgb(h) for h in hexes]
    n = len(rgbs)
    # For each x, interpolate between stops
    img = Image.new("RGB", (width, height))
    pix = img.load()
    for x in range(width):
        # t in [0,1]
        t = x / (width - 1) if width > 1 else 0
        # segment
        seg_f = t * (n - 1)
        seg = int(seg_f)
        seg = min(seg, n - 2)
        local_t = seg_f - seg
        r1, g1, b1 = rgbs[seg]
        r2, g2, b2 = rgbs[seg + 1]
        r = int(round(r1 + (r2 - r1) * local_t))
        g = int(round(g1 + (g2 - g1) * local_t))
        b = int(round(b1 + (b2 - b1) * local_t))
        for y in range(height):
            pix[x, y] = (r, g, b)
    return img


def _extract_via_kmeans(pixels: np.ndarray, n_colors: int):
    """Try sklearn KMeans, return centers (n,3) and counts (n,) or None."""
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return None, None
    try:
        # pixels shape (10000,3) uint8 -> float
        kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=0)
        labels = kmeans.fit_predict(pixels.astype(np.float32))
        centers = kmeans.cluster_centers_.astype(int)  # (n,3)
        # Clamp
        centers = np.clip(centers, 0, 255)
        counts = np.bincount(labels, minlength=n_colors)
        return centers, counts
    except Exception:
        return None, None


def _extract_via_mediancut(image: Image.Image, n_colors: int):
    """Fallback via Pillow quantize."""
    # Quantize original (not resized) ? spec says image.quantize with MEDIANCUT
    # We'll quantize a 100×100 copy for speed.
    small = image.resize((100, 100), Image.BILINEAR)
    if small.mode != "RGB":
        small = small.convert("RGB")
    try:
        q = small.quantize(colors=n_colors, method=Image.MEDIANCUT)
        # palette
        pal = q.getpalette()[: n_colors * 3]
        # q is P mode index image; count frequencies
        # Get data
        arr = np.array(q)  # (100,100) indices
        # pal may have less than n_colors*3 if colors less? pad
        centers = []
        for i in range(n_colors):
            if i * 3 + 2 < len(pal):
                centers.append((pal[i * 3], pal[i * 3 + 1], pal[i * 3 + 2]))
            else:
                centers.append((0, 0, 0))
        centers = np.array(centers, dtype=int)
        # Counts via bincount on arr
        flat = arr.flatten()
        counts = np.bincount(flat, minlength=n_colors)
        # Ensure counts align with centers order — pal order is already index order
        return centers, counts
    except Exception:
        # Last resort: sample dominant colors via numpy unique
        small_np = np.array(small).reshape(-1, 3)
        uniq, counts = np.unique(small_np, axis=0, return_counts=True)
        # take top n_colors
        idx = np.argsort(counts)[::-1][:n_colors]
        centers = uniq[idx]
        counts_sorted = counts[idx]
        # pad if needed
        if len(centers) < n_colors:
            pad_n = n_colors - len(centers)
            centers = np.vstack([centers, np.zeros((pad_n, 3), dtype=int)])
            counts_sorted = np.concatenate([counts_sorted, np.zeros(pad_n, dtype=int)])
        return centers, counts_sorted


def extract_palette(
    image: Image.Image,
    n_colors: int = 5,  # 3–8
    mode: str = "vibrant",  # vibrant | muted | pastel | deep
) -> dict:  # {hex: [str], rgb: [(r,g,b)], counts: [int], css_vars: str, gradient: PIL.Image}
    """
    Extract dominant palette from image.
    """
    n_colors = _clamp_n(n_colors)
    mode = mode.lower() if isinstance(mode, str) else "vibrant"
    if mode not in ("vibrant", "muted", "pastel", "deep"):
        mode = "vibrant"

    # Ensure RGB
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # 1. Resize to 100×100 BILINEAR, reshape(-1,3)
    small = image.resize((100, 100), Image.BILINEAR)
    arr = np.array(small).reshape(-1, 3)  # (10000,3)

    # 2. KMeans or fallback
    centers, counts = _extract_via_kmeans(arr, n_colors)
    if centers is None:
        centers, counts = _extract_via_mediancut(image, n_colors)

    # 3. Sort by counts descending (dominant first)
    # centers: (n,3), counts: (n,)
    # Ensure numpy arrays
    centers = np.array(centers)
    counts = np.array(counts)
    order = np.argsort(counts)[::-1]
    centers = centers[order]
    counts = counts[order]

    # 4. Mode transforms
    transformed = []
    for (r, g, b) in centers:
        r2, g2, b2 = _transform_color(int(r), int(g), int(b), mode)
        transformed.append((r2, g2, b2))

    hexes = [_rgb_to_hex(r, g, b) for (r, g, b) in transformed]
    rgbs = transformed  # list of tuples
    counts_list = counts.tolist()

    # 5. css_vars
    css_vars = _css_vars(hexes)

    # 6. gradient
    gradient = _generate_gradient(hexes, 800, 120)

    # 7. WCAG contrast for dominant vs white/black (computed lazily but we include)
    # We return contrasts as well for convenience
    dominant = hexes[0] if hexes else "#000000"
    contrast_white = contrast_ratio(dominant, "#ffffff")
    contrast_black = contrast_ratio(dominant, "#000000")

    # Determine pass/fail badge WCAG AA requires 4.5:1 for normal text
    def _badge(ratio: float) -> str:
        if ratio >= 7.0:
            return "AAA"
        elif ratio >= 4.5:
            return "AA"
        elif ratio >= 3.0:
            return "AA Large"
        else:
            return "Fail"

    wcag = {
        "dominant": dominant,
        "white_ratio": round(contrast_white, 2),
        "black_ratio": round(contrast_black, 2),
        "white_badge": _badge(contrast_white),
        "black_badge": _badge(contrast_black),
    }

    return {
        "hex": hexes,
        "rgb": rgbs,
        "counts": counts_list,
        "css_vars": css_vars,
        "gradient": gradient,
        "wcag": wcag,
        # also include raw centers for debugging
        "n_colors": n_colors,
        "mode": mode,
    }
