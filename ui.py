"""Single-window layout for the live translator.

The translator used to open four OS windows — camera, skeleton, filter mosaic, neuron
grid — that had to be dragged into position on every run, and which OpenCV decorates
with a Qt toolbar on Linux builds. Everything is placed in one frame here instead.

The styling is deliberately plain: near-black background, OpenCV's default font, and a
small set of roles rather than decoration. This is a debugging surface for a model, and
it should look like one.

Everything works in **BGR**, because that is what `cv2.imshow` expects.
`preprocessing.render_skeleton` returns RGB, so the caller converts once on the way in.

The module holds no camera or model code, so the whole layout can be rendered from a
synthetic state and checked as a PNG without a webcam.

## Palette

Colour is assigned by role, not chosen per-call. Every hex below is lifted directly from
the dataviz skill's validated reference palette (`references/palette.md`) rather than
picked by eye, and the three that can appear on screen together -- GOOD, WARNING, INFO --
were re-run through `validate_palette.js` as a set: CVD separation, normal-vision
separation and contrast against a #0d0d0d surface all pass. (Its categorical
lightness-band check flags WARNING as brighter than a *series* hue should be, but that
check is scoped to categorical palettes; WARNING is a status colour here, validated
against its own contrast target instead, and status colours are supposed to stand out.)

- **INK** (WHITE / GREY / MUTED) -- the dark-mode ink ramp: primary, secondary, muted.
- **GOOD** / **WARNING** -- the skill's fixed, unthemed status pair: a confident reading,
  and progress not yet committed.
- **INFO** -- the sequential ramp's default hue (blue), reused as the one accent colour
  for "the AI is looking here": the hand-focus box on the camera feed and the hidden-layer
  heatmap (INFO_DIM at the low end, INFO at the high end). One hue for one job, rather
  than the mismatched magenta box and unrelated cyan grid this replaced.
- **BLACK** stays available for grid separators and the like, where the point is maximum
  contrast against light content, not a themed role.

The finger colours in the skeleton view are a different thing entirely: transcribed
byte-for-byte from MediaPipe's own drawing style so the render matches what the model was
trained on. They are correctness, not decoration, and this palette does not touch them.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

BLACK = (0, 0, 0)
BG = (13, 13, 13)             # #0d0d0d -- page plane
PANEL = (25, 26, 26)          # #1a1a19 -- elevated surface (skeleton/filter boxes)
WHITE = (255, 255, 255)       # #ffffff -- primary ink
GREY = (183, 194, 195)        # #c3c2b7 -- secondary ink
MUTED = (129, 135, 137)       # #898781 -- muted ink / disabled
DIM = (53, 56, 56)            # #383835 -- hairline borders, unfilled bar tracks
GREEN = (12, 163, 12)         # #0ca30c -- status: good
AMBER = (25, 178, 250)        # #fab219 -- status: warning / pending
INFO = (229, 135, 57)         # #3987e5 -- accent: AI attention (focus box, hot end of the heatmap)
INFO_DIM = (107, 54, 13)      # #0d366b -- accent, dim end of the heatmap

FONT = cv2.FONT_HERSHEY_SIMPLEX

# OpenCV's Hershey fonts are ASCII-only: anything else is drawn as a literal '?'. The
# prose in this project uses em dashes and ellipses freely, so rather than relying on
# every caller remembering, strings are folded down where text is actually drawn.
_FALLBACKS = str.maketrans({
    "—": "-", "–": "-", "…": "...", "·": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'", "×": "x",
})


@dataclass
class Layout:
    camera_w: int = 800
    camera_h: int = 600
    rail_w: int = 300
    vis_w: int = 340
    bottom_h: int = 96
    pad: int = 12
    show_vis: bool = False

    @property
    def width(self):
        return self.camera_w + self.rail_w + (self.vis_w if self.show_vis else 0)

    @property
    def height(self):
        return self.camera_h + self.bottom_h


@dataclass
class ViewState:
    """Everything the layout needs for one frame."""

    camera: np.ndarray = None
    skeleton: np.ndarray = None        # BGR, already converted from render_skeleton
    scores: np.ndarray = None
    class_names: list = field(default_factory=list)
    letter: str = ""
    confidence: float = 0.0
    hand_present: bool = False
    moving: bool = False
    uncertain: bool = False
    text: str = ""
    progress: float = 0.0
    conv: np.ndarray = None
    dense: np.ndarray = None
    debug: list = field(default_factory=list)

    # The two popup buttons in the bottom bar.
    filters_enabled: bool = False    # model has named activation taps to show
    reference_enabled: bool = False  # the reference image was found on disk
    filters_open: bool = False
    reference_open: bool = False
    hover: str = None                # "filters", "reference", or None


def text(img, string, x, y, scale=0.45, color=WHITE, thickness=1):
    string = str(string).translate(_FALLBACKS).encode("ascii", "replace").decode("ascii")
    cv2.putText(img, string, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def fit_into(image, width, height, fill=None):
    """Scale preserving aspect ratio, letterboxed on `fill` (defaults to BG)."""
    canvas = np.full((height, width, 3), fill or BG, dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas

    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x, y = (width - new_w) // 2, (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def to_gray(values):
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return ((values - low) / (high - low) * 255).astype(np.uint8)


def lerp_bgr(low, high, t):
    """One-hue sequential ramp: `t` in [0, 255] -> a BGR tuple between `low` and `high`.

    Used for the hidden-layer heatmap (INFO_DIM -> INFO), so magnitude reads as one
    consistent accent colour getting brighter, the reference palette's sequential
    convention, rather than an arbitrary hue picked per project.
    """
    t = t / 255.0
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(low, high))


def draw_skeleton(canvas, state, x, y, width, size):
    text(canvas, "AI view (skeleton)", x, y, scale=0.42, color=GREY)
    y += 10
    cv2.rectangle(canvas, (x, y), (x + size, y + size), PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + size, y + size), DIM, 1)
    if state.skeleton is not None:
        canvas[y + 1:y + size, x + 1:x + size] = fit_into(state.skeleton, size - 1, size - 1, fill=PANEL)
    return y + size + 20


def draw_top3(canvas, state, x, y, width):
    text(canvas, "Top 3", x, y, scale=0.42, color=GREY)
    y += 8

    if state.scores is None or not len(state.class_names):
        return y + 3 * 26

    bar_x = x + 24
    bar_w = width - 24 - 44

    for rank, index in enumerate(np.argsort(state.scores)[::-1][:3]):
        probability = float(state.scores[index])
        row_y = y + rank * 26

        color = GREEN if rank == 0 and probability > 0.5 else GREY
        text(canvas, state.class_names[index].upper(), x, row_y + 15,
             scale=0.55, color=WHITE if rank == 0 else GREY)

        cv2.rectangle(canvas, (bar_x, row_y + 4), (bar_x + bar_w, row_y + 16), DIM, 1)
        filled = int(bar_w * probability)
        if filled > 1:
            cv2.rectangle(canvas, (bar_x, row_y + 4), (bar_x + filled, row_y + 16), color, -1)
        text(canvas, f"{probability * 100:.0f}%", bar_x + bar_w + 8, row_y + 15,
             scale=0.42, color=WHITE if rank == 0 else GREY)

    return y + 3 * 26 + 14


DEBUG_LINE_H = 15


def debug_height(lines):
    """Height `draw_debug` will take, so the rail can be budgeted before drawing."""
    return 0 if not lines else 10 + DEBUG_LINE_H * len(lines) + 14


def draw_debug(canvas, lines, x, y):
    if not lines:
        return y
    text(canvas, "Motion", x, y, scale=0.42, color=GREY)
    y += 10
    for i, line in enumerate(lines):
        text(canvas, line, x, y + 12 + i * DEBUG_LINE_H, scale=0.38, color=INFO)
    return y + DEBUG_LINE_H * len(lines) + 14


FILTER_GRID = (8, 4)   # rows, cols -- 32 filters, tall to suit a dedicated side column


def _filter_mosaic(activations, rows, cols):
    """Grayscale mosaic of conv1's feature maps, before resizing. Shared by the inline
    panel and the zoomed popup so the two can never show subtly different pictures."""
    height, cell_w, filters = activations.shape
    mosaic = np.zeros((height * rows, cell_w * cols), dtype=np.uint8)
    for i in range(min(filters, rows * cols)):
        feature = activations[:, :, i].copy()
        # Convolution padding lights up the border regardless of input; blanking it stops
        # the per-cell normalisation from being dominated by an artefact.
        feature[:2, :] = feature[-2:, :] = 0
        feature[:, :2] = feature[:, -2:] = 0
        r, c = divmod(i, cols)
        mosaic[r * height:(r + 1) * height, c * cell_w:(c + 1) * cell_w] = to_gray(feature)
    return mosaic


def _draw_mosaic_grid(canvas, x, y, width, height, rows, cols):
    """Lines between filter tiles, so a large mosaic reads as N separate filters
    rather than one sheet."""
    for r in range(1, rows):
        yy = y + height * r // rows
        cv2.line(canvas, (x, yy), (x + width, yy), BLACK, 1)
    for c in range(1, cols):
        xx = x + width * c // cols
        cv2.line(canvas, (xx, y), (xx, y + height), BLACK, 1)


def draw_filters(canvas, activations, x, y, width, pane_h):
    """`pane_h` is handed in rather than derived from `width`, so the mosaic can claim
    whatever vertical space `compose()` finds left over in the column instead of being
    capped to a fixed 2:1 aspect. 32 small tiles read poorly; this is what fixes that."""
    text(canvas, "Conv filters", x, y, scale=0.42, color=GREY)
    y += 10

    rows, cols = FILTER_GRID
    if activations is None:
        cv2.rectangle(canvas, (x, y), (x + width, y + pane_h), PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + pane_h), DIM, 1)
        return y + pane_h + 20

    mosaic = _filter_mosaic(activations, rows, cols)
    resized = cv2.resize(mosaic, (width, pane_h), interpolation=cv2.INTER_NEAREST)
    canvas[y:y + pane_h, x:x + width] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    _draw_mosaic_grid(canvas, x, y, width, pane_h, rows, cols)

    return y + pane_h + 20


def large_filter_mosaic(activations, cell_size=105):
    """Standalone, full-window version of the mosaic, for the 'Filters' zoom popup.

    Sized from FILTER_GRID's own aspect (8 rows x 4 cols, twice as tall as wide) rather
    than a fixed square -- squeezing that into a square canvas is what stretched every
    filter tile to 2x its true width in an earlier version of this function.

    No title text -- the popup's own OS window title serves that purpose.
    """
    rows, cols = FILTER_GRID
    width, height = cols * cell_size, rows * cell_size
    if activations is None:
        return np.full((height, width, 3), BG, dtype=np.uint8)

    mosaic = _filter_mosaic(activations, rows, cols)
    resized = cv2.resize(mosaic, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    _draw_mosaic_grid(canvas, 0, 0, width, height, rows, cols)
    return canvas


def neuron_grid_height(width):
    """Content height `draw_neurons` will take up, so it can be budgeted before drawing."""
    cols, rows, gap = 16, 8, 2
    cell = (width - (cols - 1) * gap) // cols
    return rows * cell + (rows - 1) * gap


def draw_neurons(canvas, dense, x, y, width):
    text(canvas, "Hidden layer (128)", x, y, scale=0.42, color=GREY)
    y += 10

    cols, rows, gap = 16, 8, 2
    cell = (width - (cols - 1) * gap) // cols
    intensities = to_gray(dense) if dense is not None else np.zeros(128, np.uint8)

    for i in range(128):
        r, c = divmod(i, cols)
        cx, cy = x + c * (cell + gap), y + r * (cell + gap)
        value = int(intensities[i])
        color = DIM if dense is None else lerp_bgr(INFO_DIM, INFO, value)
        cv2.rectangle(canvas, (cx, cy), (cx + cell, cy + cell), color, -1)

    return y + rows * cell + (rows - 1) * gap + 20


BUTTON_W = 150
BUTTON_H = 28
BUTTON_GAP = 8


def button_rects(layout):
    """Fixed positions for the two popup buttons, bottom-right of the window.

    `layout` is set once from CLI args and never changes during a run (the window is
    WINDOW_AUTOSIZE, not resizable), so these are the same every frame and safe to
    recompute cheaply wherever they're needed rather than threading them through state.
    """
    top = layout.camera_h
    x2 = layout.width - layout.pad
    y1 = top + 38
    y2 = y1 + BUTTON_H
    filters = (x2 - BUTTON_W, y1, x2, y2)
    reference = (x2 - 2 * BUTTON_W - BUTTON_GAP, y1, x2 - BUTTON_W - BUTTON_GAP, y2)
    return {"filters": filters, "reference": reference}


def hit_test(layout, px, py):
    """Which button, if any, contains point (px, py). Used by the mouse callback."""
    for name, (x1, y1, x2, y2) in button_rects(layout).items():
        if x1 <= px < x2 and y1 <= py < y2:
            return name
    return None


def draw_button(canvas, rect, label, enabled, hover, active):
    x1, y1, x2, y2 = rect
    if not enabled:
        bg, fg, border = BG, MUTED, DIM
    elif active:
        bg, fg, border = lerp_bgr(PANEL, GREEN, 60), GREEN, GREEN
    elif hover:
        bg, fg, border = lerp_bgr(PANEL, WHITE, 30), WHITE, WHITE
    else:
        bg, fg, border = PANEL, GREY, DIM

    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.4, 1)
    text(canvas, label, x1 + max(4, (x2 - x1 - tw) // 2), y1 + (y2 - y1 + th) // 2,
         scale=0.4, color=fg)


def draw_bottom(canvas, state, layout):
    top = layout.camera_h
    pad = layout.pad

    cv2.line(canvas, (0, top), (layout.width, top), DIM, 1)

    if not state.hand_present:
        reading, color = "Looking for a hand...", GREY
    elif state.uncertain:
        reading, color = "Not sure...", GREY
    else:
        reading = f"Sign: {state.letter.upper()} ({state.confidence * 100:.1f}%)"
        if state.moving:
            reading += "   [moving]"
        color = GREEN

    text(canvas, reading, pad, top + 28, scale=0.75, color=color, thickness=2)

    # Progress toward committing the current letter.
    if state.progress > 0:
        bar_w = 200
        bar_x = layout.width - pad - bar_w
        cv2.rectangle(canvas, (bar_x, top + 16), (bar_x + bar_w, top + 26), DIM, 1)
        filled = int(bar_w * min(state.progress, 1.0))
        if filled > 1:
            cv2.rectangle(canvas, (bar_x, top + 16), (bar_x + filled, top + 26), AMBER, -1)

    rects = button_rects(layout)

    # Trimmed to fit the pixel width actually left before the buttons, not a fixed
    # character count -- a 40-char run of a wide letter like W would otherwise overlap
    # the Reference button (measured: 782px of text against a button starting at 780px
    # in the narrowest window this layout produces).
    available_w = rects["reference"][0] - pad - 12
    shown = state.text
    while shown and cv2.getTextSize(shown, FONT, 0.8, 2)[0][0] > available_w:
        shown = shown[1:]
    text(canvas, shown or "-", pad, top + 62, scale=0.8, color=WHITE, thickness=2)

    text(canvas, "q quit   space   backspace   c clear   r repeat", pad,
         layout.height - 12, scale=0.4, color=GREY)

    draw_button(canvas, rects["reference"], "Reference",
                enabled=state.reference_enabled, hover=state.hover == "reference",
                active=state.reference_open)
    draw_button(canvas, rects["filters"], "Filters (zoom)",
                enabled=state.filters_enabled, hover=state.hover == "filters",
                active=state.filters_open)


TOP3_HEIGHT = 8 + 3 * 26 + 14


def compose(state, layout):
    """Build the single frame that goes to the one window."""
    canvas = np.full((layout.height, layout.width, 3), BG, dtype=np.uint8)
    canvas[:layout.camera_h, :layout.camera_w] = fit_into(
        state.camera, layout.camera_w, layout.camera_h)

    rail_x = layout.camera_w + layout.pad
    rail_w = layout.rail_w - 2 * layout.pad
    y = layout.pad + 10

    # The skeleton takes whatever the rail has left once the panels below are accounted
    # for. Sizing it from the remainder is what stops those panels from running off the
    # bottom of the column. draw_skeleton adds a fixed 30px of its own (10px title +
    # 20px trailing gap) around whatever size it's given, so that 30 is subtracted here
    # too -- verified against the real draw functions, not just this arithmetic, since
    # an earlier version of this formula silently overflowed by exactly the amount it
    # was short.
    #
    # The hidden-layer grid lives in the rail now, under Top 3, so the side column can
    # go entirely to the filter mosaic instead of splitting it with the grid.
    neuron_block = 10 + neuron_grid_height(rail_w) + 20 if layout.show_vis else 0
    reserved = TOP3_HEIGHT + neuron_block + debug_height(state.debug)
    size = max(110, min(rail_w, layout.camera_h - y - layout.pad - reserved - 30))

    y = draw_skeleton(canvas, state, rail_x, y, rail_w, size)
    y = draw_top3(canvas, state, rail_x, y, rail_w)
    if layout.show_vis:
        y = draw_neurons(canvas, state.dense, rail_x, y, rail_w)
    draw_debug(canvas, state.debug, rail_x, y)

    if layout.show_vis:
        vis_x = layout.camera_w + layout.rail_w + layout.pad
        vis_w = layout.vis_w - 2 * layout.pad
        vis_y = layout.pad + 10

        # The whole column, minus the title and trailing gap draw_filters adds around
        # whatever height it's given, which lands exactly on camera_h - pad.
        filters_h = (layout.camera_h - layout.pad) - vis_y - 30
        draw_filters(canvas, state.conv, vis_x, vis_y, vis_w, filters_h)

    cv2.line(canvas, (layout.camera_w, 0), (layout.camera_w, layout.camera_h), DIM, 1)
    if layout.show_vis:
        split = layout.camera_w + layout.rail_w
        cv2.line(canvas, (split, 0), (split, layout.camera_h), DIM, 1)

    draw_bottom(canvas, state, layout)
    return canvas
