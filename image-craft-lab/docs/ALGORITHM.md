# Algorithm Notes — Image Craft Lab

Three studios, one shared image, no ML. All transforms are pure Pillow + numpy (+ optional sklearn). This doc explains *why* each constant matters.

---

## 1. ASCII — luminance, aspect, charset

### Luminance

```python
lum = image.convert("L")   # Pillow ITU-R 601: 0.299R + 0.587G + 0.114B
```

Pillow's `convert("L")` uses **ITU-R 601** luma. The alternative often quoted for sRGB linear light is `0.2126R + 0.7152G + 0.0722B` (Rec. 709). Difference is ~10% on greens — Pillow gives slightly more weight to red/blue. We document both and use Pillow's native path (fast, no numpy).

If you need exact Rec. 709, replace with:

```python
arr = np.array(image)  # (h,w,3)
lum = (0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]).astype(np.uint8)
lum = Image.fromarray(lum, mode="L")
```

Visual difference is subtle; charset choice dominates.

### Aspect correction

Monospace glyphs are not square. A typical cell is ~1.8× taller than wide (width ≈ 0.55 × height). Without correction, output looks vertically stretched.

```
rows = int(cols * (h / w) * CHAR_ASPECT)   # CHAR_ASPECT = 0.55
```

Example: 600×400 portrait with `cols=120` → `rows = 120 * (400/600) * 0.55 ≈ 44`. Without `0.55`, rows would be 80 and the face would look squashed.

We resize luminance with `BILINEAR` (smooth) then quantize:

```
charset[int(p/255 * (len(charset)-1))]
```

`invert` applies `ImageOps.invert` on the `L` image before resize, so `" "` and `"@"` swap.

### Charsets

```
CLASSIC  = "@%#*+=-:. "          # balanced ramp, best general
BLOCKS   = "█▓▒░ "               # 4-step + space, posterized
MINIMAL  = ".:-=+*#%@"           # inverse ramp — dots to solids
DETAILED = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "  # 70 chars, photo-real
BW       = "█ "                  # threshold at 128 → hard black/white
```

More chars ≠ always better. `DETAILED` wins on high-res `cols=180` but looks noisy at `cols=40`. `BLOCKS` is most readable on thumbnails.

### Color modes

- `grayscale` — `text` only, HTML is monochrome `<pre>`.
- `bw` — threshold `p < 128 ? charset[0] : charset[-1]`.
- `original` — per-char `<span style="color:rgb(r,g,b)">` where `r,g,b` are sampled from the (contrast/brightness-adjusted) RGB image resized with `BILINEAR` to `cols×rows`. Spaces become `&#160;` to preserve layout.

### Contrast / Brightness

`ImageEnhance.Contrast/Brightness(...).enhance(factor)` with `factor` 0.5–2.0 is applied **before** `convert("L")`, so both luminance and color sampling share the enhancement. Range is clamped; values outside clip silently.

---

## 2. Pixel — NEAREST, quantize, dither

### Why NEAREST

Pixel art needs **hard edges**. The whole point is to see the grid.

1. Preserve aspect: `small_w = grid`, `small_h = int(grid * h / w)`.
2. Downscale with `Image.NEAREST` → each output pixel is a single nearest source pixel, no blending.
3. Quantize to `colors` via:

```python
small.quantize(colors=colors, method=Image.MEDIANCUT, dither=1 if dither else 0)
```

`MEDIANCUT` (Heckbert) recursively splits the RGB cube at the median along the longest axis until `colors` boxes remain. It's deterministic, fast (no iteration), and matches the `sklearn`-free path for palette.

`BILINEAR` or `LANCZOS` would anti-alias the downscale and erase the block look — never use them for the pixel path.

After quantize, we `convert("RGB")` then upscale with `NEAREST × scale` (crisp blocks). `scale` is display-only (e.g. 8×) — the logical `small` is `grid×small_h`.

### Dither

Pillow's `dither=1` enables Floyd-Steinberg error diffusion *during* quantize. At low `colors` (4–8) it trades banding for noise; at 32+ the difference is subtle. We expose it as a toggle. No custom kernel — Pillow's built-in only (per "Out — do not build" scope).

### CSS box-shadow

For fun — and as a shareable artifact without an image:

```python
css = ", ".join(f"{x*scale}px {y*scale}px 0 #{r:02x}{g:02x}{b:02x}" for y,x,r,g,b in pixels)
# .pixel-art { width: 8px; height: 8px; box-shadow: <css> }
```

A `32×32` sprite → 1,024 shadows (~20 KB CSS). A `128×128` sprite → 16k shadows (~400 KB) — we generate it but warn in UI that copying >64× is heavy. The CSS renders with a single `<div>` — useful for codepen / blog embeds.

### Grid vs. recognition

Empirically: a face remains a face down to `grid=32` (with 16 colors). At `16×16`, only high-contrast features survive. That's why presets start at 32.

---

## 3. Palette — KMeans vs MEDIANCUT, HSV transforms, WCAG

### Resize & clustering

```
small = image.resize((100,100), BILINEAR)  # 10,000 pixels
arr = np.array(small).reshape(-1,3)
```

100×100 is enough to represent the image's color distribution while keeping KMeans cheap (10k points × 5–8 centers × ~10 iterations).

### KMeans path (if sklearn available)

```python
KMeans(n_clusters=n_colors, n_init=5, random_state=0).fit(arr)
centers = cluster_centers_  # (n,3)
counts = bincount(labels)   # (n,) — sorted dominant-first
```

- `n_init=5` runs 5 random restarts and keeps best inertia — avoids poor local minima on images with skewed palettes.
- `random_state=0` makes palette deterministic for same input → cacheable and testable.
- Dominant sort (`counts` descending) puts the most common color first — that's the one used for WCAG vs white/black and for gradient start.

### MEDIANCUT fallback (no sklearn)

```python
q = small.quantize(colors=n_colors, method=Image.MEDIANCUT)
palette = q.getpalette()[:n_colors*3]  # indexed
counts = bincount(np.array(q).flatten())
```

Same `MEDIANCUT` as pixel path. Result is very close to KMeans on natural photos (earth tones, skin, sky) but differs on synthetics with isolated saturated pixels — KMeans isolates them as a cluster; MEDIANCUT may merge them if they're spatially rare. Documented both paths; `README` tells user `pip install scikit-learn` is optional.

### HSV mode transforms

We use `colorsys.rgb_to_hsv` → adjust → `hsv_to_rgb`, with channels in `[0,1]`:

- `vibrant` — no change.
- `muted` — `s * 0.60` (desaturate).
- `pastel` — `s * 0.50`, `v = min(1, v + 0.25)` (wash + brighten). Classic pastel recipe.
- `deep` — `v * 0.70` (darken/value crush). Keeps hue/saturation but lowers brightness — good for "moody" variants.

All outputs clamped `0–255`. Differences are tested: `vibrant` vs `pastel` must differ on at least one hex.

### CSS vars + Tailwind

```
:root { --color-1: #hex; ... }
colors: { "color-1": "#hex", ... }
```

Returned as `css_vars` string for one-click copy. User can paste `:root` into any stylesheet.

### Gradient

800×120 linear interpolation between sorted hexes. For each `x`:

```
t = x/(W-1), seg = int(t*(n-1)), local_t = frac
rgb = lerp(rgb[seg], rgb[seg+1], local_t)
```

Wallpaper-ready PNG export (no text, just color).

### WCAG contrast

Relative luminance per channel:

```
c_srgb = c/12.92                if c <= 0.04045
         ((c+0.055)/1.055)^2.4  otherwise
L = 0.2126*R + 0.7152*G + 0.0722*B
ratio = (L_lighter + 0.05) / (L_darker + 0.05)
```

We compute `ratio` for dominant hex vs `#ffffff` and vs `#000000`:

- `≥7.0` → AAA, `≥4.5` → AA, `≥3.0` → AA Large, else Fail.
- Badges shown inline; they guide whether white/black text over the dominant swatch passes WCAG.

Example: dominant `#1a1a2e` vs white ≈ 15.2:1 (AAA), vs black ≈ 1.38:1 (Fail) → use white text.

---

## 4. Caching & performance

- `@st.cache_data` per engine keyed on `(image_bytes_hash, all_params)`. Same image + same preset → instant (no recompute) when switching tabs.
- Live preset thumbnails use reduced params (`64-col`, `32-grid`, `5-swatch`) — cheap to compute, still representative.
- All three studios read the **same** `PIL.Image` from `image_bytes_hash` — upload once.

## 5. Limits & roadmap

- No video/webcam (would need temporal dithering).
- No hand-paint canvas drag (deferred to v0.2.0 — documented in PROJECT_BRIEF).
- ASCII `.png` export uses `PIL.ImageDraw` with `DejaVuSansMono` fallback — not pixel-perfect to HTML but faithful for sharing.
- For CLI parity, a future `python -m src.ascii input.jpg --preset classic --cols 120 > out.txt` is planned but not shipped.

---

*Build small. Keep the loop local.*
