"""
Presets & constants for all three studios.
Copy verbatim charsets as specified in PROJECT_BRIEF.md.
"""

CHARSET_CLASSIC = "@%#*+=-:. "
CHARSET_BLOCKS = "█▓▒░ "
CHARSET_MINIMAL = ".:-=+*#%@"
CHARSET_DETAILED = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
CHARSET_BW = "█ "
CHAR_ASPECT = 0.55

# ---- ASCII presets (6 + Full Control) ----
ASCII_PRESETS = {
    "Classic Mono": {
        "charset": CHARSET_CLASSIC,
        "charset_name": "CLASSIC",
        "cols": 120,
        "color_mode": "grayscale",
        "contrast": 1.0,
        "brightness": 1.0,
        "invert": False,
    },
    "Dense Blocks": {
        "charset": CHARSET_BLOCKS,
        "charset_name": "BLOCKS",
        "cols": 100,
        "color_mode": "grayscale",
        "contrast": 1.2,
        "brightness": 1.0,
        "invert": False,
    },
    "Minimal Dots": {
        "charset": CHARSET_MINIMAL,
        "charset_name": "MINIMAL",
        "cols": 140,
        "color_mode": "grayscale",
        "contrast": 1.0,
        "brightness": 1.0,
        "invert": False,
    },
    "Original Color": {
        "charset": CHARSET_CLASSIC,
        "charset_name": "CLASSIC",
        "cols": 120,
        "color_mode": "original",
        "contrast": 1.0,
        "brightness": 1.0,
        "invert": False,
    },
    "High Contrast BW": {
        "charset": CHARSET_BW,
        "charset_name": "BW",
        "cols": 100,
        "color_mode": "bw",
        "contrast": 1.5,
        "brightness": 1.0,
        "invert": False,
    },
    "Detailed": {
        "charset": CHARSET_DETAILED,
        "charset_name": "DETAILED",
        "cols": 180,
        "color_mode": "grayscale",
        "contrast": 1.0,
        "brightness": 1.0,
        "invert": False,
    },
}

# For UI: ordered list
ASCII_PRESET_ORDER = [
    "Classic Mono",
    "Dense Blocks",
    "Minimal Dots",
    "Original Color",
    "High Contrast BW",
    "Detailed",
]

CHARSET_MAP = {
    "CLASSIC": CHARSET_CLASSIC,
    "BLOCKS": CHARSET_BLOCKS,
    "MINIMAL": CHARSET_MINIMAL,
    "DETAILED": CHARSET_DETAILED,
    "BW": CHARSET_BW,
}

# ---- Pixel presets (4 + Full Control) ----
PIXEL_PRESETS = {
    "Tiny Sprite": {"grid": 32, "colors": 16, "dither": False, "scale": 8},
    "GB Boy": {"grid": 64, "colors": 4, "dither": False, "scale": 8},
    "Crisp HD": {"grid": 96, "colors": 32, "dither": False, "scale": 8},
    "Dithered": {"grid": 64, "colors": 16, "dither": True, "scale": 8},
}

PIXEL_PRESET_ORDER = ["Tiny Sprite", "GB Boy", "Crisp HD", "Dithered"]

# ---- Palette presets ----
# Mode variants; n_colors slider is Full Control (3–8)
PALETTE_MODES = ["vibrant", "muted", "pastel", "deep"]
PALETTE_MODE_LABELS = {
    "vibrant": "Vibrant",
    "muted": "Muted",
    "pastel": "Pastel",
    "deep": "Deep",
}

# Valid ranges (documented / enforced)
ASCII_COLS_MIN, ASCII_COLS_MAX = 40, 300
PIXEL_GRID_CHOICES = [16, 32, 64, 96, 128]
PIXEL_COLOR_CHOICES = [4, 8, 16, 32, 64]
PALETTE_N_MIN, PALETTE_N_MAX = 3, 8
