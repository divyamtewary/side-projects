"""
Image Craft Lab — one photo, three studios.
Streamlit app: ASCII / PIXEL / PALETTE with live preset thumbnails.

Run: streamlit run app.py
"""

import hashlib
import io
import html
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.ascii import image_to_ascii
from src.palette import extract_palette, contrast_ratio
from src.pixel import image_to_pixel
from src.presets import (
    ASCII_PRESET_ORDER,
    ASCII_PRESETS,
    CHAR_ASPECT,
    CHARSET_MAP,
    PALETTE_MODE_LABELS,
    PALETTE_MODES,
    PIXEL_PRESET_ORDER,
    PIXEL_PRESETS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Image Craft Lab",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme / CSS (dark, matches side-projects banner)
# ---------------------------------------------------------------------------
BG_DEEP = "#090A16"
BG_CARD = "#17132D"
ACCENT = "#FF6B9A"
ACCENT2 = "#5EEAD4"
ACCENT3 = "#FFB86B"
TEXT1 = "#F7F5FF"
TEXT2 = "#B8B7D1"
TEXT3 = "#8E8BA8"
BORDER = "rgba(180,156,255,0.14)"
BORDER_LIT = "rgba(255,107,154,0.28)"

st.markdown(
    f"""
<style>
.stApp {{
    background:
        radial-gradient(circle at top right, rgba(255,107,154,0.10), transparent 28%),
        radial-gradient(circle at bottom left, rgba(94,234,212,0.08), transparent 24%),
        linear-gradient(180deg, #070712 0%, {BG_DEEP} 42%, #060610 100%);
}}
header[data-testid="stHeader"] {{ background: transparent !important; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
div.block-container {{ padding-top: 1rem; padding-bottom: 1.6rem; max-width: 1280px; }}
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(23,19,45,0.88);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 6px;
    gap: 6px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 12px;
    color: {TEXT2};
    font-weight: 600;
    font-size: 13px;
    padding: 8px 16px;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(90deg, rgba(255,107,154,0.18) 0%, rgba(94,234,212,0.14) 100%) !important;
    border: 1px solid {BORDER_LIT} !important;
    color: {TEXT1} !important;
}}
div[data-testid="stMetric"] {{
    background: linear-gradient(180deg, rgba(24,20,40,0.96) 0%, rgba(18,16,32,0.96) 100%);
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 12px;
}}
.glass {{
    background: linear-gradient(180deg, rgba(24,20,40,0.95) 0%, rgba(16,14,28,0.95) 100%);
    border: 1px solid {BORDER};
    border-radius: 22px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 18px 48px rgba(0,0,0,0.25);
}}
.hero-panel {{
    padding: 20px 24px;
    border-radius: 22px;
    border: 1px solid {BORDER};
    background: linear-gradient(135deg, rgba(24,20,45,0.98) 0%, rgba(15,14,28,0.98) 100%);
    box-shadow: 0 24px 64px rgba(0,0,0,0.3);
    margin-bottom: 14px;
}}
.deck-kicker {{
    color: {ACCENT};
    font-size: 11px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.35rem;
}}
.deck-title {{
    color: {TEXT1};
    font-size: 1.9rem;
    line-height: 1.08;
    font-weight: 800;
    margin: 0;
}}
.deck-subtitle {{
    color: {TEXT2};
    font-size: 0.95rem;
    line-height: 1.65;
    margin-top: 0.45rem;
}}
.stat-chip {{
    background: rgba(255,255,255,0.04);
    border: 1px solid {BORDER};
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 12px;
    color: {TEXT2};
}}
.stat-chip b {{ color: {TEXT1}; }}
.preset-card {{
    background: linear-gradient(180deg, rgba(26,22,42,0.96) 0%, rgba(18,16,32,0.96) 100%);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}}
.preset-card:hover {{ border-color: {BORDER_LIT}; }}
.preset-card.active {{ border-color: {ACCENT}; box-shadow: 0 0 0 1px rgba(255,107,154,0.25); }}
.swatch {{
    border-radius: 14px;
    padding: 10px 12px;
    color: #fff;
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 12px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.45);
}}
.wcag-pass {{ background: rgba(34,197,94,0.18); color: #86efac; border: 1px solid rgba(34,197,94,0.35); border-radius: 999px; padding: 2px 8px; font-size: 11px; font-weight: 700; }}
.wcag-fail {{ background: rgba(239,68,68,0.14); color: #fca5a5; border: 1px solid rgba(239,68,68,0.30); border-radius: 999px; padding: 2px 8px; font-size: 11px; font-weight: 700; }}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 4px; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def image_bytes_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def load_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    # Convert to RGB if needed, preserve for caching
    if img.mode == "P":
        img = img.convert("RGB")
    return img


def ascii_to_png(text: str, font_size: int = 12, bg: str = "#0a0a0f", fg: str = "#f5f7fb") -> Image.Image:
    """Render ASCII text to PNG for download."""
    lines = text.split("\n")
    if not lines:
        lines = [""]
    cols = max(len(line) for line in lines)
    rows = len(lines)
    # Estimate char cell: ~7px wide at 12px monospace, ~14px tall
    # Try to load a monospace font, fallback to default
    try:
        # DejaVu Sans Mono is common
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("consola.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
    # Measure char
    try:
        # bbox for "M"
        bbox = font.getbbox("M")
        cw = bbox[2] - bbox[0]
        ch = bbox[3] - bbox[1]
        if cw <= 0 or ch <= 0:
            cw, ch = 7, 14
    except Exception:
        cw, ch = 7, 14
    # Add a bit of leading
    ch = int(ch * 1.1) + 2
    cw = max(6, cw)
    pad = 16
    W = cols * cw + pad * 2
    H = rows * ch + pad * 2
    # Cap huge images
    max_w, max_h = 3000, 4000
    if W > max_w:
        W = max_w
    if H > max_h:
        H = max_h
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    y = pad
    for line in lines:
        draw.text((pad, y), line, fill=fg, font=font)
        y += ch
        if y > H - pad:
            break
    return img


def get_sample_paths():
    base = Path(__file__).parent / "assets" / "samples"
    return {
        "Portrait": base / "portrait.jpg",
        "Landscape": base / "landscape.jpg",
        "High Contrast": base / "high_contrast.jpg",
    }


# ---------------------------------------------------------------------------
# Caching — per engine keyed on (hash, params)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_ascii(image_hash: str, image_bytes: bytes, cols: int, charset_name: str, color_mode: str, contrast: float, brightness: float, invert: bool):
    img = load_image_from_bytes(image_bytes)
    charset = CHARSET_MAP.get(charset_name, CHARSET_MAP["CLASSIC"])
    return image_to_ascii(img, cols=cols, charset=charset, color_mode=color_mode, contrast=contrast, brightness=brightness, invert=invert)


@st.cache_data(show_spinner=False)
def cached_pixel(image_hash: str, image_bytes: bytes, grid: int, colors: int, dither: bool, scale: int):
    img = load_image_from_bytes(image_bytes)
    return image_to_pixel(img, grid=grid, colors=colors, dither=dither, scale=scale)


@st.cache_data(show_spinner=False)
def cached_palette(image_hash: str, image_bytes: bytes, n_colors: int, mode: str):
    img = load_image_from_bytes(image_bytes)
    return extract_palette(img, n_colors=n_colors, mode=mode)


# ---------------------------------------------------------------------------
# Session init
# ---------------------------------------------------------------------------
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "image_hash" not in st.session_state:
    st.session_state.image_hash = None
if "image_name" not in st.session_state:
    st.session_state.image_name = None
if "ascii_preset" not in st.session_state:
    st.session_state.ascii_preset = "Classic Mono"
if "pixel_preset" not in st.session_state:
    st.session_state.pixel_preset = "Tiny Sprite"
if "palette_mode" not in st.session_state:
    st.session_state.palette_mode = "vibrant"
if "palette_n" not in st.session_state:
    st.session_state.palette_n = 5
# Full Control states
if "ascii_cols" not in st.session_state:
    st.session_state.ascii_cols = 120
if "ascii_charset" not in st.session_state:
    st.session_state.ascii_charset = "CLASSIC"
if "ascii_color" not in st.session_state:
    st.session_state.ascii_color = "grayscale"
if "ascii_contrast" not in st.session_state:
    st.session_state.ascii_contrast = 1.0
if "ascii_brightness" not in st.session_state:
    st.session_state.ascii_brightness = 1.0
if "ascii_invert" not in st.session_state:
    st.session_state.ascii_invert = False
if "pixel_grid" not in st.session_state:
    st.session_state.pixel_grid = 64
if "pixel_colors" not in st.session_state:
    st.session_state.pixel_colors = 16
if "pixel_dither" not in st.session_state:
    st.session_state.pixel_dither = False
if "pixel_scale" not in st.session_state:
    st.session_state.pixel_scale = 8

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
# Banner — show svg if exists (robust to Pillow / SVG)
banner_path = Path(__file__).parent / "docs" / "img" / "banner.svg"
if banner_path.exists():
    try:
        if banner_path.suffix.lower() == ".svg":
            svg_text = banner_path.read_text(encoding="utf-8")
            # Streamlit can render raw SVG via markdown; ensure it fills container
            st.markdown(svg_text, unsafe_allow_html=True)
        else:
            st.image(str(banner_path), use_container_width=True)
    except Exception:
        st.image(str(banner_path), use_container_width=True)
else:
    st.markdown(
        f"""
<div class="hero-panel">
  <div class="deck-kicker">image craft lab — one photo, three studios</div>
  <div class="deck-title">ASCII • PIXEL • PALETTE</div>
  <div class="deck-subtitle">Upload one photo. Every preset shows a live thumbnail of <i>that</i> photo. Pure Pillow — no model, no API, runs on anything.</div>
</div>
""",
        unsafe_allow_html=True,
    )

# Hero text fallback if banner shown but still want subtitle
st.markdown(
    f"""
<div style="margin: -6px 0 14px 0; color:{TEXT2}; font-size: 0.95rem; line-height: 1.6;">
Upload one photo. Three studios share it: <b style="color:{TEXT1}">ASCII</b> (text) • <b style="color:{TEXT1}">PIXEL</b> (sprite) • <b style="color:{TEXT1}">PALETTE</b> (colors). Each has presets with live thumbnails on <i>that</i> photo + Full Control. No ML — just Pillow.
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Shared — upload + samples
# ---------------------------------------------------------------------------
left, right = st.columns([1.15, 0.85], gap="medium")

with left:
    st.markdown(f'<div style="color:{TEXT1}; font-weight:700; font-size: 0.95rem; margin-bottom: 6px;">Upload photo</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload photo",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        help="Max 10 MB. PNG / JPG / JPEG / WEBP.",
    )
    if uploaded is not None:
        data = uploaded.getvalue()
        if len(data) > 10 * 1024 * 1024:
            st.error("File too large — max 10 MB.")
        else:
            # Validate image
            try:
                Image.open(io.BytesIO(data)).verify()
                st.session_state.image_bytes = data
                st.session_state.image_hash = image_bytes_hash(data)
                st.session_state.image_name = uploaded.name
            except Exception as e:
                st.error(f"Could not read image: {e}")

with right:
    st.markdown(f'<div style="color:{TEXT1}; font-weight:700; font-size: 0.95rem; margin-bottom: 6px;">Or try a sample</div>', unsafe_allow_html=True)
    samples = get_sample_paths()
    # Show 3 columns with thumbnails + Use sample buttons
    scols = st.columns(3)
    for idx, (label, path) in enumerate(samples.items()):
        with scols[idx]:
            if path.exists():
                # show thumbnail
                try:
                    thumb = Image.open(path)
                    st.image(thumb, use_container_width=True, caption=label)
                except Exception:
                    st.caption(label)
            else:
                st.caption(label)
            if st.button(f"Use {label}", key=f"sample_{label}", use_container_width=True):
                try:
                    data = path.read_bytes()
                    st.session_state.image_bytes = data
                    st.session_state.image_hash = image_bytes_hash(data)
                    st.session_state.image_name = path.name
                    st.rerun()
                except Exception as e:
                    st.error(f"Cannot load sample: {e}")
    st.caption("Samples are synthetic (Pillow-generated) — no licences needed. Portraits are illustrated, not photos.")

# Shared thumbnail + filename + size row
if st.session_state.image_bytes is not None:
    img = load_image_from_bytes(st.session_state.image_bytes)
    w, h = img.size
    size_kb = len(st.session_state.image_bytes) / 1024
    c1, c2 = st.columns([0.22, 0.78])
    with c1:
        st.image(img, caption="Original", use_container_width=True)
    with c2:
        st.markdown(
            f"""
<div class="glass" style="margin-top: 0;">
  <div style="color:{TEXT1}; font-weight:700;">{html.escape(st.session_state.image_name or 'image')}</div>
  <div style="color:{TEXT2}; font-size: 0.92rem; margin-top: 4px;">{w} × {h} px • {size_kb:.1f} KB • hash <code>{st.session_state.image_hash[:8]}</code></div>
  <div style="margin-top: 10px; display:flex; gap:8px; flex-wrap:wrap;">
    <span class="stat-chip">CHAR_ASPECT <b>0.55</b></span>
    <span class="stat-chip">NEAREST <b>pixel</b></span>
    <span class="stat-chip">KMeans <b>or MEDIANCUT</b></span>
  </div>
  <div style="color:{TEXT3}; font-size: 0.82rem; margin-top: 8px;">Same bytes feed all three studios. Preset cards below are live thumbnails of <i>this</i> image.</div>
</div>
""",
            unsafe_allow_html=True,
        )
else:
    st.info("↑ Upload a photo or click **Use sample** to start. The three studios share the same image — switch tabs without re-uploading.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
if st.session_state.image_bytes is None:
    st.stop()

# Global vars for caching
image_bytes = st.session_state.image_bytes
image_hash = st.session_state.image_hash

tab_ascii, tab_pixel, tab_palette = st.tabs(["◐  ASCII", "▦  PIXEL", "◎  PALETTE"])

# ===========================================================================
# ASCII TAB
# ===========================================================================
with tab_ascii:
    st.markdown(
        f'<div class="deck-kicker">studio a — ascii art</div><div style="color:{TEXT2}; font-size: 0.92rem; margin-bottom: 10px;">Text mosaics from luminance. 6 presets + Full Control. Live thumbs at 64 cols.</div>',
        unsafe_allow_html=True,
    )

    # Live preset previews — 2×3 grid (3 cols × 2 rows)
    # Row 1
    preset_cols = st.columns(3)
    thumb_cols = 64  # for live thumb
    for i, preset_name in enumerate(ASCII_PRESET_ORDER):
        col = preset_cols[i % 3]
        with col:
            preset = ASCII_PRESETS[preset_name]
            # cached thumb
            thumb = cached_ascii(
                image_hash,
                image_bytes,
                64,
                preset["charset_name"],
                preset["color_mode"],
                preset["contrast"],
                preset["brightness"],
                preset["invert"],
            )
            is_active = st.session_state.ascii_preset == preset_name
            border_cls = "active" if is_active else ""
            # Card
            st.markdown(
                f'<div class="preset-card {border_cls}"><div style="color:{TEXT1}; font-weight:700; font-size: 0.92rem;">{preset_name}</div><div style="color:{TEXT3}; font-size: 0.78rem; margin-top: 2px;">{preset["charset_name"]} • {preset["cols"]} cols • {preset["color_mode"]} • c{preset["contrast"]}</div></div>',
                unsafe_allow_html=True,
            )
            # Thumb preview — show as <pre> scrolled
            # For original color, thumb html has colors; but for thumbnail we show plain text to save space
            txt_thumb = thumb["text"]
            # Truncate lines for thumb display: show first ~18 rows, cols already 64
            thumb_lines = txt_thumb.split("\n")[:22]
            thumb_text = "\n".join(thumb_lines)
            st.code(thumb_text, language=None)
            if st.button(
                "Select" if not is_active else "✓ Selected",
                key=f"ascii_sel_{preset_name}",
                use_container_width=True,
                disabled=is_active,
            ):
                st.session_state.ascii_preset = preset_name
                # Sync Full Control sliders to preset
                st.session_state.ascii_cols = preset["cols"]
                st.session_state.ascii_charset = preset["charset_name"]
                st.session_state.ascii_color = preset["color_mode"]
                st.session_state.ascii_contrast = preset["contrast"]
                st.session_state.ascii_brightness = preset["brightness"]
                st.session_state.ascii_invert = preset["invert"]
                st.rerun()
        # after 3, new row
        if i == 2:
            preset_cols = st.columns(3)

    # Full Control expander — also when a preset is active, show its params as editable?
    # Add a radio to switch to Full Control mode
    st.markdown("---")
    use_full = st.checkbox("Full Control — override preset params", value=False, key="ascii_full_toggle")
    if use_full:
        st.markdown(f'<div style="color:{TEXT1}; font-weight:700; margin-bottom: 6px;">Full Control</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.session_state.ascii_cols = st.slider("Columns", 40, 300, st.session_state.ascii_cols, 10, key="ascii_cols_slider")
            st.session_state.ascii_charset = st.selectbox("Charset", list(CHARSET_MAP.keys()), index=list(CHARSET_MAP.keys()).index(st.session_state.ascii_charset), key="ascii_charset_sel")
        with fc2:
            st.session_state.ascii_color = st.radio("Color mode", ["grayscale", "original", "bw"], index=["grayscale", "original", "bw"].index(st.session_state.ascii_color), horizontal=True, key="ascii_color_radio")
            st.session_state.ascii_invert = st.checkbox("Invert", value=st.session_state.ascii_invert, key="ascii_invert_chk")
        with fc3:
            st.session_state.ascii_contrast = st.slider("Contrast", 0.5, 2.0, st.session_state.ascii_contrast, 0.1, key="ascii_contrast_slider")
            st.session_state.ascii_brightness = st.slider("Brightness", 0.5, 2.0, st.session_state.ascii_brightness, 0.1, key="ascii_brightness_slider")

        # Use Full Control params for main preview
        main_cols = st.session_state.ascii_cols
        main_charset_name = st.session_state.ascii_charset
        main_color = st.session_state.ascii_color
        main_contrast = st.session_state.ascii_contrast
        main_brightness = st.session_state.ascii_brightness
        main_invert = st.session_state.ascii_invert
        main_label = f"Full Control — {main_charset_name} • {main_cols} cols • {main_color}"
    else:
        preset = ASCII_PRESETS[st.session_state.ascii_preset]
        main_cols = preset["cols"]
        main_charset_name = preset["charset_name"]
        main_color = preset["color_mode"]
        main_contrast = preset["contrast"]
        main_brightness = preset["brightness"]
        main_invert = preset["invert"]
        main_label = st.session_state.ascii_preset

    # Main preview
    main = cached_ascii(image_hash, image_bytes, main_cols, main_charset_name, main_color, main_contrast, main_brightness, main_invert)
    st.markdown(f'<div style="color:{TEXT1}; font-weight:700; margin-top: 8px;">Main preview — {html.escape(main_label)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{TEXT3}; font-size: 0.82rem;">{main["width"]} × {main["height"]} • {main["chars"]} chars • CHAR_ASPECT 0.55 • {main_charset_name} • {main_color}</div>', unsafe_allow_html=True)

    # Stats row
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Columns × Rows", f"{main['width']} × {main['height']}")
    sc2.metric("Total chars", f"{main['chars']:,}")
    sc3.metric("Charset", main_charset_name)

    # Preview rendering
    if main_color == "original":
        # HTML with colors
        st.markdown(main["html"], unsafe_allow_html=True)
    else:
        # Use code block with monospace, small font via markdown html
        # Also provide plain text area
        st.markdown(
            f'<pre style="line-height:1.0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 7px; background:#0a0a0f; color:#f5f7fb; padding:12px; border-radius:12px; overflow:auto; white-space:pre; border:1px solid {BORDER};">{html.escape(main["text"])}</pre>',
            unsafe_allow_html=True,
        )

    # Downloads
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download .txt",
            data=main["text"].encode("utf-8"),
            file_name="ascii.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download .html",
            data=main["html"].encode("utf-8"),
            file_name="ascii.html",
            mime="text/html",
            use_container_width=True,
        )
    with d3:
        # PNG export
        png_img = ascii_to_png(main["text"])
        buf = io.BytesIO()
        png_img.save(buf, format="PNG")
        st.download_button(
            "Download .png",
            data=buf.getvalue(),
            file_name="ascii.png",
            mime="image/png",
            use_container_width=True,
        )
    with st.expander("Copy text / html"):
        st.code(main["text"][:8000], language=None)
        st.code(main["html"][:8000], language="html")

# ===========================================================================
# PIXEL TAB
# ===========================================================================
with tab_pixel:
    st.markdown(
        f'<div class="deck-kicker">studio b — pixel art</div><div style="color:{TEXT2}; font-size: 0.92rem; margin-bottom: 10px;">Hard-edge NEAREST downscale → MEDIANCUT quantize → NEAREST upscale. Live thumbs at 32-grid.</div>',
        unsafe_allow_html=True,
    )

    # Preset cards 2×2
    pcols = st.columns(2)
    for i, preset_name in enumerate(PIXEL_PRESET_ORDER):
        col = pcols[i % 2]
        with col:
            preset = PIXEL_PRESETS[preset_name]
            thumb = cached_pixel(image_hash, image_bytes, 32, preset["colors"], preset["dither"], 8)
            is_active = st.session_state.pixel_preset == preset_name
            border_cls = "active" if is_active else ""
            st.markdown(
                f'<div class="preset-card {border_cls}"><div style="color:{TEXT1}; font-weight:700;">{preset_name}</div><div style="color:{TEXT3}; font-size: 0.78rem;">{preset["grid"]}× • {preset["colors"]} colors • dither {"on" if preset["dither"] else "off"}</div></div>',
                unsafe_allow_html=True,
            )
            st.image(thumb["image"], use_container_width=True)
            st.caption(f'thumb 32→{thumb["small_w"]}×{thumb["small_h"]} • css {len(thumb["css"])//1000}k chars')
            if st.button("Select" if not is_active else "✓ Selected", key=f"pixel_sel_{preset_name}", use_container_width=True, disabled=is_active):
                st.session_state.pixel_preset = preset_name
                st.session_state.pixel_grid = preset["grid"]
                st.session_state.pixel_colors = preset["colors"]
                st.session_state.pixel_dither = preset["dither"]
                st.session_state.pixel_scale = preset["scale"]
                st.rerun()
        if i == 1:
            pcols = st.columns(2)

    st.markdown("---")
    use_full_pix = st.checkbox("Full Control — override preset", value=False, key="pixel_full_toggle")
    if use_full_pix:
        st.markdown(f'<div style="color:{TEXT1}; font-weight:700; margin-bottom: 6px;">Full Control</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.session_state.pixel_grid = st.select_slider("Grid", options=[16, 32, 64, 96, 128], value=st.session_state.pixel_grid, key="pixel_grid_slider")
        with fc2:
            st.session_state.pixel_colors = st.select_slider("Colors", options=[4, 8, 16, 32, 64], value=st.session_state.pixel_colors, key="pixel_colors_slider")
        with fc3:
            st.session_state.pixel_dither = st.toggle("Dither", value=st.session_state.pixel_dither, key="pixel_dither_toggle")
            st.session_state.pixel_scale = st.slider("Scale (export)", 4, 16, st.session_state.pixel_scale, 1, key="pixel_scale_slider")
        main_grid = st.session_state.pixel_grid
        main_colors = st.session_state.pixel_colors
        main_dither = st.session_state.pixel_dither
        main_scale = st.session_state.pixel_scale
        main_pix_label = f"Full Control — {main_grid}× • {main_colors}c • dither {'on' if main_dither else 'off'}"
    else:
        preset = PIXEL_PRESETS[st.session_state.pixel_preset]
        main_grid = preset["grid"]
        main_colors = preset["colors"]
        main_dither = preset["dither"]
        main_scale = preset["scale"]
        main_pix_label = st.session_state.pixel_preset

    main_pix = cached_pixel(image_hash, image_bytes, main_grid, main_colors, main_dither, main_scale)
    st.markdown(f'<div style="color:{TEXT1}; font-weight:700; margin-top: 8px;">Main preview — {html.escape(main_pix_label)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{TEXT3}; font-size: 0.82rem;">{main_pix["small_w"]} × {main_pix["small_h"]} → {main_pix["width"]} × {main_pix["height"]} • {main_pix["colors"]} colors • dither {"on" if main_dither else "off"} • NEAREST</div>', unsafe_allow_html=True)

    # Stats
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Grid", f"{main_pix['small_w']} × {main_pix['small_h']}")
    sc2.metric("Export", f"{main_pix['width']} × {main_pix['height']}")
    sc3.metric("Colors", f"{main_pix['colors']}")

    st.image(main_pix["image"], caption=f"Pixel — {main_pix_label}", use_container_width=True)

    # CSS box-shadow preview
    with st.expander("CSS box-shadow (shareable)"):
        st.caption("Copy this CSS to render the pixel art with a single div + box-shadow (no image).")
        css_code = f".pixel-art {{ width: {main_scale}px; height: {main_scale}px; background: transparent; box-shadow: {main_pix['css']}; }}"
        st.code(css_code[:12000] + ("\n/* ... truncated ... */" if len(css_code) > 12000 else ""), language="css")
        # Download CSS
        st.download_button("Download CSS", data=css_code.encode("utf-8"), file_name="pixel.css", mime="text/css", use_container_width=True)

    # Downloads
    d1, d2 = st.columns(2)
    with d1:
        buf = io.BytesIO()
        main_pix["image"].save(buf, format="PNG")
        st.download_button("Download PNG", data=buf.getvalue(), file_name="pixel.png", mime="image/png", use_container_width=True)
    with d2:
        # Also download small quantized as PNG
        buf2 = io.BytesIO()
        main_pix["small"].save(buf2, format="PNG")
        st.download_button("Download small PNG (grid)", data=buf2.getvalue(), file_name="pixel_small.png", mime="image/png", use_container_width=True)


# ===========================================================================
# PALETTE TAB
# ===========================================================================
with tab_palette:
    st.markdown(
        f'<div class="deck-kicker">studio c — palette</div><div style="color:{TEXT2}; font-size: 0.92rem; margin-bottom: 10px;">KMeans or MEDIANCUT → HSV transforms. WCAG contrast badges included.</div>',
        unsafe_allow_html=True,
    )

    # Mode pills + n_colors slider
    mcols = st.columns([3, 1])
    with mcols[0]:
        # Mode as radio horizontal
        mode = st.radio("Mode", PALETTE_MODES, index=PALETTE_MODES.index(st.session_state.palette_mode), horizontal=True, format_func=lambda m: PALETTE_MODE_LABELS[m], key="palette_mode_radio")
        st.session_state.palette_mode = mode
    with mcols[1]:
        n = st.slider("Colors", 3, 8, st.session_state.palette_n, 1, key="palette_n_slider")
        st.session_state.palette_n = n

    # Live palette preview — show swatches for current mode/n
    # Also show tiny preview of all modes? Spec says mode pills + n slider; live thumbs 5-swatch.
    # We'll show main palette below, but also small row of mode variants thumbnails
    st.markdown(f'<div style="color:{TEXT3}; font-size: 0.82rem; margin-top: 4px;">Live thumbnails — 5-swatch preview per mode (click pill to switch main view)</div>', unsafe_allow_html=True)
    # Thumbnail row: 4 modes at n=5
    tcols = st.columns(4)
    for i, m in enumerate(PALETTE_MODES):
        with tcols[i]:
            thumb_pal = cached_palette(image_hash, image_bytes, 5, m)
            # Render 5 swatches mini
            swatch_html = "".join(
                f'<div style="flex:1; height:28px; background:{h}; border-radius:6px; border:1px solid rgba(255,255,255,0.12);"></div>'
                for h in thumb_pal["hex"]
            )
            st.markdown(
                f'<div class="preset-card" style="padding:8px;"><div style="color:{TEXT1}; font-size: 0.82rem; font-weight:700; text-align:center;">{PALETTE_MODE_LABELS[m]}</div><div style="display:flex; gap:4px; margin-top:6px;">{swatch_html}</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(" ".join(thumb_pal["hex"][:3]))

    # Main palette
    pal = cached_palette(image_hash, image_bytes, st.session_state.palette_n, st.session_state.palette_mode)
    st.markdown("---")
    st.markdown(f'<div style="color:{TEXT1}; font-weight:700;">Swatches — {PALETTE_MODE_LABELS[pal["mode"]]} • {pal["n_colors"]} colors</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{TEXT3}; font-size: 0.82rem;">Sorted dominant first. KMeans if sklearn else MEDIANCUT. HSV transforms: muted s×0.6, pastel s×0.5 v+0.25, deep v×0.7.</div>', unsafe_allow_html=True)

    # Swatches with hex+rgb+copy + counts
    # Use columns per swatch
    sw_cols = st.columns(len(pal["hex"]))
    for idx, (hex_c, rgb_c, cnt) in enumerate(zip(pal["hex"], pal["rgb"], pal["counts"])):
        with sw_cols[idx]:
            # Luminance for text color choice
            r, g, b = rgb_c
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            text_col = "#0a0a0f" if lum > 150 else "#ffffff"
            pct = (cnt / sum(pal["counts"]) * 100) if sum(pal["counts"]) else 0
            st.markdown(
                f'<div class="swatch" style="background:{hex_c}; color:{text_col}; border:1px solid rgba(255,255,255,0.14); text-align:center;">'
                f'<div style="font-weight:800; font-size: 0.95rem;">{hex_c}</div>'
                f'<div style="font-size: 0.78rem; opacity:0.9;">rgb({r},{g},{b})</div>'
                f'<div style="font-size: 0.72rem; margin-top:4px; opacity:0.85;">{cnt} • {pct:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Copy helper via st.code (native copy)
            st.code(hex_c, language=None)

    # Gradient preview + WCAG badges
    st.markdown(f'<div style="color:{TEXT1}; font-weight:700; margin-top: 12px;">Gradient wallpaper</div>', unsafe_allow_html=True)
    st.image(pal["gradient"], caption="800×120 linear gradient of the palette", use_container_width=True)

    # WCAG contrast
    wcag = pal["wcag"]
    dominant = wcag["dominant"]
    st.markdown(f'<div style="color:{TEXT2}; font-size: 0.92rem; margin-top: 8px;">Dominant <code>{dominant}</code> contrast:</div>', unsafe_allow_html=True)
    w1, w2 = st.columns(2)
    with w1:
        badge_cls = "wcag-pass" if wcag["white_badge"] != "Fail" else "wcag-fail"
        ratio = wcag["white_ratio"]
        badge = wcag["white_badge"]
        st.markdown(
            f'<div class="glass" style="display:flex; justify-content:space-between; align-items:center; padding:12px 16px;">'
            f'<span style="color:{TEXT1};">vs <b style="background:#fff; color:#000; padding:2px 6px; border-radius:6px;">white #ffffff</b> — {ratio}:1</span>'
            f'<span class="{badge_cls}">{badge}</span></div>',
            unsafe_allow_html=True,
        )
        # also show ratio details
        st.caption(f"WCAG AA needs ≥4.5:1, AAA ≥7:1. Dominant vs white is {badge}.")
    with w2:
        badge_cls = "wcag-pass" if wcag["black_badge"] != "Fail" else "wcag-fail"
        ratio = wcag["black_ratio"]
        badge = wcag["black_badge"]
        st.markdown(
            f'<div class="glass" style="display:flex; justify-content:space-between; align-items:center; padding:12px 16px;">'
            f'<span style="color:{TEXT1};">vs <b style="background:#000; color:#fff; padding:2px 6px; border-radius:6px;">black #000000</b> — {ratio}:1</span>'
            f'<span class="{badge_cls}">{badge}</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Dominant vs black is {badge}.")

    # Copy CSS / Download gradient PNG
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div style="color:{TEXT1}; font-weight:700; margin-bottom: 6px;">CSS variables</div>', unsafe_allow_html=True)
        st.code(pal["css_vars"], language="css")
        st.download_button("Download CSS", data=pal["css_vars"].encode("utf-8"), file_name="palette.css", mime="text/css", use_container_width=True)
    with c2:
        st.markdown(f'<div style="color:{TEXT1}; font-weight:700; margin-bottom: 6px;">Export gradient</div>', unsafe_allow_html=True)
        buf = io.BytesIO()
        pal["gradient"].save(buf, format="PNG")
        st.download_button("Download gradient PNG", data=buf.getvalue(), file_name="gradient.png", mime="image/png", use_container_width=True)
        # Also show tailwind snippet hint
        st.caption("CSS includes `:root --color-*` + Tailwind `colors` snippet.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f'<div style="text-align:center; color:{TEXT3}; font-size: 0.82rem;">'
    f'Image Craft Lab • Pillow • numpy • optional scikit-learn • Streamlit • '
    f'<a href="https://github.com/divyamtewary/side-projects" style="color:{ACCENT};">side-projects</a> • MIT • '
    f'No upload leaves your browser except to Streamlit server memory.</div>',
    unsafe_allow_html=True,
)
