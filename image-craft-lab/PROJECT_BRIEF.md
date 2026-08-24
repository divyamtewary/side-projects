# Project — `image-craft-lab`

**Repo:** `Side Projects/image-craft-lab` *(renamed from `ascii-art-lab` — incorporates ASCII + Pixel + Palette)*
**Session slot:** Day 2, 15:30 → 19:00 (**210 minutes**)
**AI credit cap:** **$1.20**
**Narrative role:** *Play.* One photo, three transformations. The most shareable lab in the repo — beginners see results in 5 seconds, intermediates get full control.
**Build status:** **READY — build exactly as specified.**

> User uploads **one** photo. Three studios share it: **ASCII** (text), **PIXEL** (sprite), **PALETTE** (colors). Each has presets with live thumbnails on *that* photo + a Full Control mode. No model, no API, pure Pillow — runs on anything. This merges the three photo-upload ideas into one polished, general-purpose `image-craft-lab` as requested.

---

## 1. Resources

| Resource | Role |
|---|---|
| `Pillow>=10.0` | Load, resize, `ImageEnhance`, `ImageOps`, `ImageDraw` for all three studios |
| `numpy>=1.26` | Luminance / color arrays, fast quantisation |
| `scikit-learn>=1.3` | `KMeans` for palette extraction — **optional**. Provide pure-Pillow fallback (`quantize(method=MEDIANCUT)`) so `pip install scikit-learn` is not required. Document both paths. |
| `streamlit>=1.36` | UI — matches `Side Projects/neural-observatory/app.py`. One app, three tabs. |
| `docs/img/banner.svg` | Copy dark banner style from `Side Projects/docs/img/banner.svg` |

No ML, no torch, no token, no weights. `Pillow` is the only hard dep.

---

## 2. Scope

### In — ship all three studios in one app

#### 2.1 Shared — upload + samples (build this first)

- `st.file_uploader(["png","jpg","jpeg","webp"], max 10MB)` + original thumbnail.
- `assets/samples/` — 3 permissive images (portrait, landscape, high-contrast) with `Use sample` buttons so demo works without upload. Never commit copyrighted images without licence.
- Global `image_bytes_hash` for `@st.cache_data` keys. All three studios read the same `PIL.Image`.

#### 2.2 Studio A — ASCII Art (`src/ascii.py`)

Pure functions, no UI imports:

```python
def image_to_ascii(
    image: PIL.Image.Image,
    cols: int = 120,
    charset: str = "@%#*+=-:. ",
    color_mode: str = "original",  # original | grayscale | bw
    contrast: float = 1.0,          # 0.5–2.0
    brightness: float = 1.0,
    invert: bool = False,
) -> dict:  # {text, html, width, height, chars}
```

- Algorithm (exact):
  1. `ImageEnhance.Contrast/Brightness`, `ImageOps.invert` on L if needed.
  2. `convert("L")` for luminance (document `0.2126R+0.7152G+0.0722B` vs Pillow's formula).
  3. Aspect-correct resize: `rows = int(cols * (h/w) * 0.55)` where `CHAR_ASPECT=0.55` (monospace cell ~1.8× taller).
  4. Quantize `p → charset[int(p/255*(len(charset)-1))]`.
  5. `original` → per-char `<span style="color:rgb(r,g,b)">`, `bw` → threshold 128 to 2 chars.
- `cols` 40–300, `contrast/brightness` 0.5–2.0.
- Charsets (copy verbatim):

```python
CHARSET_CLASSIC  = "@%#*+=-:. "
CHARSET_BLOCKS   = "█▓▒░ "
CHARSET_MINIMAL  = ".:-=+*#%@"
CHARSET_DETAILED = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
CHARSET_BW       = "█ "
CHAR_ASPECT      = 0.55
```

- Presets (6 + Full Control):

| Preset | charset | cols | color_mode | contrast |
|---|---|---|---|---|
| Classic Mono | CLASSIC | 120 | grayscale | 1.0 |
| Dense Blocks | BLOCKS | 100 | grayscale | 1.2 |
| Minimal Dots | MINIMAL | 140 | grayscale | 1.0 |
| Original Color | CLASSIC | 120 | original | 1.0 |
| High Contrast BW | BW | 100 | bw | 1.5 |
| Detailed | DETAILED | 180 | grayscale | 1.0 |
| Full Control | user | slider | radio | sliders |

#### 2.3 Studio B — Pixel Art (`src/pixel.py`)

```python
def image_to_pixel(
    image: PIL.Image.Image,
    grid: int = 64,            # 16/32/64/96/128 — output is grid × grid (square) or grid × scaled
    colors: int = 16,          # 8/16/32/64
    dither: bool = False,
    scale: int = 8,            # display scale factor for PNG export (e.g., 8×)
) -> dict:  # {image: PIL.Image, css: str, grid, colors}
```

- Algorithm:
  1. Preserve aspect: `small_w = grid`, `small_h = int(grid * h/w)`.
  2. Downscale with `NEAREST` to `small_w × small_h` (hard edges).
  3. Quantize to `colors` via `image.quantize(colors=colors, method=MEDIANCUT, dither=1 if dither else 0)` then `convert("RGB")`.
  4. Upscale for display/export with `NEAREST` × `scale` (crisp blocks).
  5. Generate `css` as `box-shadow` string for fun: `2px 2px 0 #hex, ...` (optional, but include — it's the shareable artifact).
- Presets (4 + Full Control):

| Preset | grid | colors | dither | Look |
|---|---|---|---|---|
| Tiny Sprite | 32 | 16 | off | 32×32 retro |
| GB Boy | 64 | 4 | off | 4-color Game Boy |
| Crisp HD | 96 | 32 | off | Detailed |
| Dithered | 64 | 16 | on | Floyd nostalgia |
| Full Control | slider | slider | toggle | all params |

#### 2.4 Studio C — Palette (`src/palette.py`)

```python
def extract_palette(
    image: PIL.Image.Image,
    n_colors: int = 5,         # 3–8
    mode: str = "vibrant",     # vibrant | muted | pastel | deep
) -> dict:  # {hex: [str], rgb: [(r,g,b)], counts: [int], css_vars: str, gradient: PIL.Image}
```

- Algorithm:
  1. Resize to 100×100 (`BILINEAR`), `reshape(-1,3)`.
  2. If `sklearn` available: `KMeans(n_clusters=n_colors, n_init=5, random_state=0)` → cluster centers + counts. Else: `image.quantize(colors=n_colors, method=MEDIANCUT)` → palette.
  3. Sort by `counts` descending (dominant first).
  4. `mode` transforms: `muted` → `saturation *0.6`, `pastel` → `saturation*0.5 + lightness+0.25`, `deep` → `value*0.7`. Implement via `colorsys.rgb_to_hsv` → adjust → `hsv_to_rgb`. Clamp 0–255.
  5. Generate `css_vars`: `:root { --color-1: #hex; }` + Tailwind `colors: {}` snippet.
  6. Generate `gradient` as 800×120 linear gradient of the hexes for wallpaper export.
  7. WCAG contrast: compute `contrast_ratio(hex1, hex2)` for dominant vs white/black, show pass/fail badge.
- Presets are `mode` variants, `n_colors` slider is the Full Control.

#### 2.5 UI — `app.py` (one app, three tabs)

```
┌──────────────────────────────────────────────────┐
│ Header + banner                                  │
│ [ Upload + Use sample 3 buttons ]  (shared)      │
│ original thumbnail + filename + size             │
├──────────────────────────────────────────────────┤
│ Tabs: [ ASCII ] [ PIXEL ] [ PALETTE ]            │
│                                                  │
│ ASCII tab:  2×3 preset cards (live thumbs)       │
│             + Full Control expander              │
│             + main preview (monospace) + stats   │
│             + Download .txt / .html / .png       │
│                                                  │
│ PIXEL tab:  2×2 preset cards (pixel thumbs)      │
│             + Full Control (grid/colors/dither)  │
│             + main pixel preview (scaled)        │
│             + Download PNG / CSS box-shadow      │
│                                                  │
│ PALETTE tab: mode pills + n_colors slider        │
│             + swatches (hex+rgb+copy) + counts   │
│             + gradient preview + WCAG badges     │
│             + Copy CSS / Download gradient PNG   │
└──────────────────────────────────────────────────┘
```

- **Live preset previews:** Each card renders a **64-col (ASCII) / 32-grid (pixel) / 5-swatch (palette) thumbnail** of the *currently uploaded* image, cached via `@st.cache_data` keyed on `(hash, mode, params)`. This is the "preview on uploaded image" requirement — now for all three studios.
- **Caching:** `@st.cache_data` per engine keyed on `(hash, all_params)`.
- **Copy:** `st.code(css, language="css")` + native copy. Hex chips with `st.color_picker` disabled for display.
- **Stats:** ASCII `cols×rows • chars`, Pixel `grid×grid • colors • dither`, Palette `n_colors • dominant hex • contrast`.

#### 2.6 Docs & polish

- `README.md` — hero GIF (upload → tab switch → preset click → export), one-line pitch, three-studio table, algorithm notes (`CHAR_ASPECT=0.55`, `NEAREST`, KMeans vs MEDIANCUT), setup, export formats.
- `docs/ALGORITHM.md` — three sections: ASCII (luminance, aspect), Pixel (nearest, quantize, dither), Palette (k-means, HSV transforms, WCAG).
- `requirements.txt`, `LICENSE` (MIT), `.gitignore`
- `docs/img/banner.svg` + 3 screenshots `01-ascii.png`, `02-pixel.png`, `03-palette.png`

#### 2.7 Tests — `tests/test_craft.py` (≥8, no UI, generated images only)

- ASCII: black→CHARSET[0], white→CHARSET[-1], cols clamp, bw 2-char, invert swaps
- Pixel: output size = grid×scaled, colors ≤ requested, dither flag doesn't crash
- Palette: `len(hex)==n_colors`, hex format `#rrggbb`, vibrant vs pastel differ, counts sum, missing sklearn fallback still returns n_colors

### Out — do not build

- No video/webcam, no dithering Floyd-Steinberg tuning beyond Pillow's built-in, no hand-paint pixel editor (canvas drag is v0.2.0 — document as roadmap)
- No backend API (Streamlit is the app)
- No auth / gallery / accounts

---

## 3. Target structure

```
image-craft-lab/
├── PROJECT_BRIEF.md
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── app.py                          # single Streamlit app (~500 lines)
├── src/
│   ├── __init__.py
│   ├── ascii.py                    # image_to_ascii
│   ├── pixel.py                    # image_to_pixel
│   ├── palette.py                  # extract_palette + WCAG
│   └── presets.py                  # all presets + constants
├── assets/
│   └── samples/
│       ├── portrait.jpg
│       ├── landscape.jpg
│       └── high_contrast.jpg
├── docs/
│   ├── ALGORITHM.md
│   └── img/
│       ├── banner.svg
│       ├── 01-ascii.png
│       ├── 02-pixel.png
│       └── 03-palette.png
└── tests/
    └── test_craft.py
```

---

## 4. Execution plan — 210 minutes

| Clock | Minutes | Task |
|---|---|---|
| 15:30 | 15 | Scaffold tree, `requirements.txt` (`streamlit`, `pillow`, `numpy`, `scikit-learn` optional), `.gitignore`, `LICENSE`, `assets/samples/` |
| 15:45 | 35 | `src/ascii.py` + `src/presets.py` (ASCII part) — test with black/white/gradient in REPL |
| 16:20 | 30 | `src/pixel.py` — NEAREST downscale + quantize + upscale + css gen |
| 16:50 | 30 | `src/palette.py` — KMeans + MEDIANCUT fallback + HSV mode transforms + WCAG + gradient gen |
| 17:20 | 45 | `app.py` — shared upload + 3 tabs, preset card grids with live thumbnails (cache!), main previews, stats |
| 18:05 | 20 | `app.py` — Full Control expanders + exports (txt/html/png for ASCII, png+css for pixel, css+gradient png for palette) |
| 18:25 | 15 | `tests/test_craft.py` — 8 tests, `pytest -q` green |
| 18:40 | 15 | `README.md` + `docs/ALGORITHM.md` — hero screenshot, three-studio table, algorithm notes. Capture 3 screenshots |
| 18:55 | 5 | Scrub gate → commit → push → tag `craft-lab-v0.1.0` |

**If behind at 18:05:** cut ASCII `.png` export and Pixel `css box-shadow` export — keep `.txt`/`.png` core. Palette gradient PNG is 10 lines — never cut. Preset live thumbnails are the hero — never cut.

---

## 5. Setup & run

```bash
git clone https://github.com/divyamtewary/side-projects.git
cd side-projects/image-craft-lab
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt   # scikit-learn optional; without it palette uses MEDIANCUT
streamlit run app.py
# open http://localhost:8501
# Upload photo → ASCII tab → click preset card → Download .txt
# Switch to PIXEL → Tiny Sprite → Download PNG
# Switch to PALETTE → Pastel → Copy CSS
```

No weights, no token, no GPU. Works offline after `pip install`.

---

## 6. Stretch — only after 18:40

- Hand-paint pixel editor (grid canvas)
- ASCII dithering toggle
- Palette → export ASE / Coolors URL

---

## 7. Scrub gate

```powershell
git grep -i -E "indigo|interglobe|goindigo|navitaire|newskies" -- .; if ($?) { Write-Host "FAIL" }
git grep -E "C:\\Users\\|OneDrive" -- .; if ($?) { Write-Host "FAIL" }
# assets/samples licences checked
```

---

## 8. Definition of done

- [ ] `streamlit run app.py` → upload photo → **3 tabs** each show preset cards with live thumbnails of *that* photo
- [ ] ASCII: 6 presets + Full Control (charset/cols/color/contrast/brightness/invert) live-update; exports .txt/.html (and .png if not cut)
- [ ] Pixel: 4 presets + Full Control (grid/colors/dither) live-update; exports PNG + css (if not cut)
- [ ] Palette: 4 modes + n_colors 3–8 slider; swatches with hex+copy, WCAG badges, gradient preview, CSS + gradient PNG export
- [ ] 3 samples in `assets/samples/` with Use sample buttons
- [ ] `pytest -q` green (8 tests)
- [ ] `README.md` with hero GIF/screenshot, three-studio table, algorithm notes; 3 screenshots in `docs/img/` embedded
- [ ] No copyrighted samples, no absolute paths, tagged `craft-lab-v0.1.0`, pushed

---

## 9. Roadmap (document, don't build)

- Pixel hand-paint editor + onion skin
- ASCII edge mode (Canny) + video
- Palette ASE/Coolors export + palette-from-palette (harmony rules)
- CLI: `python -m src.ascii input.jpg --preset classic --cols 120 > out.txt`

---

## 10. Blog seeds

- Why `0.55` and `NEAREST` matter more than charset choice
- `MEDIANCUT` vs `KMeans` palette — when they agree and when they don't
- At what `grid` does a face remain a face?
- The most surprising palette from a boring photo
