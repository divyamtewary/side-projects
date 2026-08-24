"""
Pixel Art engine — pure functions, no Streamlit imports.

Algorithm (exact):
  1. Preserve aspect: small_w = grid, small_h = int(grid * h/w).
  2. Downscale with NEAREST to small_w × small_h (hard edges).
  3. Quantize to colors via quantize(colors=colors, method=MEDIANCUT, dither=1 if dither else 0) then convert("RGB").
  4. Upscale for display/export with NEAREST × scale (crisp blocks).
  5. Generate css as box-shadow string.
"""

from __future__ import annotations

from PIL import Image


def _clamp_grid(grid: int) -> int:
    return max(8, min(256, int(grid)))


def _clamp_colors(colors: int) -> int:
    return max(2, min(256, int(colors)))


def image_to_pixel(
    image: Image.Image,
    grid: int = 64,  # 16/32/64/96/128 — output is grid × grid (square) or grid × scaled
    colors: int = 16,  # 8/16/32/64
    dither: bool = False,
    scale: int = 8,  # display scale factor for PNG export (e.g., 8×)
) -> dict:  # {image: PIL.Image, css: str, grid, colors}
    """
    Convert PIL image to pixel art.

    Returns:
        {
            image: PIL.Image (upscaled RGB, size = small_w*scale × small_h*scale),
            small: PIL.Image (downscaled quantized, for css gen),
            css: str (box-shadow),
            grid: int,
            colors: int,
            small_w: int,
            small_h: int,
            width: int,
            height: int,
        }
    """
    grid = _clamp_grid(grid)
    colors = _clamp_colors(colors)
    scale = max(1, min(32, int(scale)))

    # Ensure RGB
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if w == 0 or h == 0:
        # fallback 1×1
        empty = Image.new("RGB", (scale, scale), (0, 0, 0))
        return {"image": empty, "small": empty, "css": "", "grid": grid, "colors": colors, "small_w": 1, "small_h": 1, "width": scale, "height": scale}

    # 1. Preserve aspect
    small_w = grid
    small_h = int(grid * h / w) if w else grid
    small_h = max(1, small_h)

    # 2. Downscale with NEAREST to small_w × small_h
    small = image.resize((small_w, small_h), Image.NEAREST)

    # 3. Quantize to colors via MEDIANCUT
    # Note: quantize expects P mode; dither=1 enables Floyd-Steinberg via Pillow's built-in
    try:
        dither_flag = 1 if dither else 0
        # quantize may fail if colors > 256 or small image; clamp
        quantized_p = small.quantize(colors=colors, method=Image.MEDIANCUT, dither=dither_flag)
        small_q = quantized_p.convert("RGB")
        # Actual colors may be less than requested if image is simple
    except Exception:
        # Fallback: no quantize
        small_q = small

    # 4. Upscale for display/export with NEAREST × scale (crisp blocks)
    large_w = small_w * scale
    large_h = small_h * scale
    large = small_q.resize((large_w, large_h), Image.NEAREST)

    # 5. Generate css as box-shadow string
    # 2px 2px 0 #hex, ... — each pixel is a shadow offset
    # For css preview we use scale as pixel size? spec says css: box-shadow string optional
    # We'll generate with 1px base and offsets = x*scale, y*scale? But more shareable is 1px blocks.
    # Simpler: generate with e.g., "x*8px y*8px 0 #hex"
    # To keep string reasonable, we cap grid for css generation to 32 if grid >32 to avoid huge string?
    # No — generate full but caller can truncate.
    css = _generate_css(small_q, scale)

    return {
        "image": large,
        "small": small_q,
        "css": css,
        "grid": grid,
        "colors": colors,
        "small_w": small_w,
        "small_h": small_h,
        "width": large_w,
        "height": large_h,
    }


def _generate_css(small_img: Image.Image, scale: int) -> str:
    """Generate box-shadow CSS from small quantized image."""
    w, h = small_img.size
    # Limit css generation to max 64*64 = 4096 shadows to avoid insane string
    # If larger, we sample? But spec grid max 128 so worst 128*~128 = 16k shadows — heavy but ok.
    # We'll generate all.
    px = small_img.load()
    parts = []
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            # offset x*scale, y*scale with blur 0 and spread 0
            # Use px units
            parts.append(f"{x*scale}px {y*scale}px 0 {hex_color}")
    return ", ".join(parts)


def pixel_css_block(small_img: Image.Image, pixel_size: int = 8) -> str:
    """
    Helper: generate a minimal CSS class for embedding.
    Returns a string like:
    .pixel-art { width: 8px; height: 8px; box-shadow: ... }
    """
    css = _generate_css(small_img, pixel_size)
    w, h = small_img.size
    return (
        f".pixel-art {{\n"
        f"  width: {pixel_size}px;\n"
        f"  height: {pixel_size}px;\n"
        f"  background: transparent;\n"
        f"  box-shadow: {css};\n"
        f"  /* canvas: {w}×{h} pixels → {w*pixel_size}×{h*pixel_size}px */\n"
        f"}}"
    )
