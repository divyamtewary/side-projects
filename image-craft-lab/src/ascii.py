"""
ASCII Art engine — pure functions, no Streamlit imports.

Algorithm (exact as in PROJECT_BRIEF.md):
  1. ImageEnhance.Contrast / Brightness, ImageOps.invert on L if needed.
  2. convert("L") for luminance (Pillow uses ITU-R 601 luma: 0.299R+0.587G+0.114B —
     documented vs alternative 0.2126R+0.7152G+0.0722B).
  3. Aspect-correct resize: rows = int(cols * (h/w) * 0.55) where CHAR_ASPECT=0.55
     (monospace cell ~1.8× taller than wide).
  4. Quantize p -> charset[int(p/255*(len(charset)-1))].
  5. original -> per-char <span style="color:rgb(r,g,b)">, bw -> threshold 128 to 2 chars.
"""

from __future__ import annotations

import html as html_lib
from PIL import Image, ImageEnhance, ImageOps

from .presets import CHAR_ASPECT, CHARSET_CLASSIC

# Clamp ranges as per spec
COLS_MIN, COLS_MAX = 40, 300
CONTRAST_MIN, CONTRAST_MAX = 0.5, 2.0
BRIGHTNESS_MIN, BRIGHTNESS_MAX = 0.5, 2.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def image_to_ascii(
    image: Image.Image,
    cols: int = 120,
    charset: str = CHARSET_CLASSIC,
    color_mode: str = "original",  # original | grayscale | bw
    contrast: float = 1.0,  # 0.5–2.0
    brightness: float = 1.0,
    invert: bool = False,
) -> dict:
    """
    Convert PIL image to ASCII art.

    Returns:
        {text, html, width, height, chars}

    - text: plain monospace string with \\n rows
    - html: <pre> friendly HTML with optional <span color>
    - width/height: cols/rows
    - chars: total characters (cols*rows)
    """
    if not charset:
        charset = CHARSET_CLASSIC
    # Clamp cols / contrast / brightness
    cols = int(_clamp(int(cols), COLS_MIN, COLS_MAX))
    contrast = _clamp(float(contrast), CONTRAST_MIN, CONTRAST_MAX)
    brightness = _clamp(float(brightness), BRIGHTNESS_MIN, BRIGHTNESS_MAX)

    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")
    # Ensure RGB for color sampling
    if image.mode == "RGBA":
        # composite over white
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    if image.mode == "L":
        # promote to RGB for color_mode handling
        image_rgb = image.convert("RGB")
    else:
        image_rgb = image.convert("RGB")

    w, h = image_rgb.size
    if w == 0 or h == 0:
        return {"text": "", "html": "", "width": 0, "height": 0, "chars": 0}

    # 1. Enhance contrast / brightness on RGB, then L for invert
    # Apply enhancements to a copy
    enhanced = image_rgb
    if contrast != 1.0:
        enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)
    if brightness != 1.0:
        enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness)

    # Prepare luminance image
    lum = enhanced.convert("L")
    if invert:
        lum = ImageOps.invert(lum)
        # also invert color source for original mode ? Brief says invert on L if needed.
        # For original mode, we invert lum only; color stays as enhanced (not inverted)
        # but if invert True we also invert the rgb for consistency in html?
        # We'll invert RGB as well logically for original color mode so colors match inverted lum.
        # However spec says "ImageOps.invert on L if needed" — so only L.
        # We keep enhanced RGB as-is for color sampling; inverting L alone flips mapping.

    # 2. Already have lum. Note luminance formula:
    # Pillow's convert("L") uses ITU-R 601: L = 0.299*R + 0.587*G + 0.114*B
    # Alternative (sRGB linear) is 0.2126R+0.7152G+0.0722B — ~10% difference on greens.

    # 3. Aspect-correct resize
    # rows = int(cols * (h/w) * CHAR_ASPECT)
    aspect = h / w if w else 1.0
    rows = int(cols * aspect * CHAR_ASPECT)
    rows = max(1, rows)

    # For color sampling we need resized RGB as well
    # Use BILINEAR for smooth lum; NEAREST would be blocky.
    lum_small = lum.resize((cols, rows), Image.BILINEAR)
    rgb_small = enhanced.resize((cols, rows), Image.BILINEAR)

    n_chars = len(charset)
    # Precompute LUT for speed
    # charset[int(p/255*(n-1))]
    lut = [charset[int(p / 255 * (n_chars - 1))] for p in range(256)]

    # 4 & 5 depending on color_mode
    color_mode = color_mode.lower() if isinstance(color_mode, str) else "original"
    if color_mode not in ("original", "grayscale", "bw"):
        color_mode = "grayscale"

    # bw threshold uses 2-char charset: darkest / lightest
    if color_mode == "bw":
        # Use charset BW logic: threshold 128 to 2 chars
        # If charset has >2 chars, use first and last.
        c_dark = charset[0]
        c_light = charset[-1]
        lines = []
        html_lines = []
        lum_px = lum_small.load()
        for y in range(rows):
            row_chars = []
            html_row_parts = []
            for x in range(cols):
                p = lum_px[x, y]
                ch = c_dark if p < 128 else c_light
                # Escape html special chars
                ch_esc = html_lib.escape(ch)
                # For bw html, no color span needed but keep pre styling
                row_chars.append(ch)
                html_row_parts.append(ch_esc)
            lines.append("".join(row_chars))
            html_lines.append("".join(html_row_parts))
        text = "\n".join(lines)
        html_body = "\n".join(html_lines)
        html = f'<pre style="line-height:1.0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 7px; margin:0; background:#0a0a0f; color:#f5f7fb; padding:12px; border-radius:12px; overflow:auto;">{html_body}</pre>'
        return {"text": text, "html": html, "width": cols, "height": rows, "chars": cols * rows}

    elif color_mode == "grayscale":
        lines = []
        lum_px = lum_small.load()
        for y in range(rows):
            row_chars = []
            for x in range(cols):
                p = lum_px[x, y]
                row_chars.append(lut[p])
            lines.append("".join(row_chars))
        text = "\n".join(lines)
        # html grayscale: no spans, just monochrome pre
        # Escape for html
        html_body = html_lib.escape(text)
        html = f'<pre style="line-height:1.0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 7px; margin:0; background:#0a0a0f; color:#f5f7fb; padding:12px; border-radius:12px; overflow:auto;">{html_body}</pre>'
        return {"text": text, "html": html, "width": cols, "height": rows, "chars": cols * rows}

    else:  # original
        lines = []
        html_lines = []
        lum_px = lum_small.load()
        rgb_px = rgb_small.load()
        for y in range(rows):
            row_chars = []
            html_row_parts = []
            for x in range(cols):
                p = lum_px[x, y]
                ch = lut[p]
                r, g, b = rgb_px[x, y]
                row_chars.append(ch)
                ch_esc = html_lib.escape(ch)
                # For spaces, preserve but still span for color consistency
                if ch == " ":
                    ch_esc = "&#160;"  # non-breaking space keeps layout
                html_row_parts.append(f'<span style="color:rgb({r},{g},{b})">{ch_esc}</span>')
            lines.append("".join(row_chars))
            html_lines.append("".join(html_row_parts))
        text = "\n".join(lines)
        html_body = "".join(f"{line}<br>" for line in html_lines)
        # Wrap in pre-like div for spans
        html = (
            '<div style="line-height:1.0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; '
            'font-size: 7px; background:#0a0a0f; color:#f5f7fb; padding:12px; border-radius:12px; overflow:auto; white-space:pre;">'
            f"{html_body}</div>"
        )
        return {"text": text, "html": html, "width": cols, "height": rows, "chars": cols * rows}
