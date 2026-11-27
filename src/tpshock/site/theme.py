"""Colour roles for the dashboard, in light and dark mode.

Every chart colour does one job:
  - ACCENT (blue): the series the story is about (usually the term premium)
  - SECOND (aqua): the one comparison series, when a chart needs two identities. Blue and aqua
    pass the colour-blindness checks in both modes (CVD Delta E 23 light / 20 dark); aqua sits
    below 3:1 contrast on the light surface, so it is always direct-labelled
  - CONTEXT (grey, the muted ink): series that give context but aren't the point (expected rates beside the
    term premium, the control group beside the meetings)
  - INK: totals and text
Blue and violet were rejected: they collapse for deuteranopes (Delta E 4.5).
Charts are built with the LIGHT values; the page swaps each one for its DARK twin in dark mode,
so every colour used in a figure must come from LIGHT.
"""

LIGHT = {
    "surface": "#fcfcfb", "page": "#f6f6f4", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)",
    "accent": "#2a78d6", "accent_band": "rgba(42,120,214,0.12)", "accent_soft": "#cde2fb",
    "second": "#1baf7a", "second_band": "rgba(27,175,122,0.12)",
    "context_band": "rgba(137,135,129,0.14)", "context_soft": "#d9d8d2",
    "good": "#0ca30c", "good_text": "#006300", "bad": "#d03b3b",
}
DARK = {
    "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)",
    "accent": "#3987e5", "accent_band": "rgba(57,135,229,0.16)", "accent_soft": "#184f95",
    "second": "#199e70", "second_band": "rgba(25,158,112,0.16)",
    "context_band": "rgba(154,152,143,0.16)", "context_soft": "#45443f",
    "good": "#0ca30c", "good_text": "#0ca30c", "bad": "#d03b3b",
}
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

# light -> dark, for the page's chart re-theming (only colours that differ)
SWAP = {LIGHT[k]: DARK[k] for k in LIGHT if LIGHT[k] != DARK[k]}
assert len(set(LIGHT.values())) == len(LIGHT), "light colours must be unique so the swap is unambiguous"
