"""Single-window layout for the live translator.

The translator used to open four OS windows — camera, skeleton, filter mosaic, neuron
grid — that had to be dragged into position on every run, and which OpenCV decorates
with a Qt toolbar on Linux builds. Everything is placed in one frame here instead.

The styling is deliberately plain: black background, OpenCV's default font, the same
white/green/grey palette the separate windows used. This is a debugging surface for a
model, and it should look like one.

Everything works in **BGR**, because that is what `cv2.imshow` expects.
`preprocessing.render_skeleton` returns RGB, so the caller converts once on the way in.

The module holds no camera or model code, so the whole layout can be rendered from a
synthetic state and checked as a PNG without a webcam.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
DIM = (70, 70, 70)
GREEN = (0, 255, 0)
AMBER = (0, 200, 255)

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
    camera_w: int = 720
    camera_h: int = 540
    rail_w: int = 260
    vis_w: int = 300
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


def text(img, string, x, y, scale=0.45, color=WHITE, thickness=1):
    string = str(string).translate(_FALLBACKS).encode("ascii", "replace").decode("ascii")
    cv2.putText(img, string, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def fit_into(image, width, height):
    """Scale preserving aspect ratio, letterboxed on black."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
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


def draw_skeleton(canvas, state, x, y, width, size):
    text(canvas, "AI view (skeleton)", x, y, scale=0.42, color=GREY)
    y += 10
    cv2.rectangle(canvas, (x, y), (x + size, y + size), DIM, 1)
    if state.skeleton is not None:
        canvas[y + 1:y + size, x + 1:x + size] = fit_into(state.skeleton, size - 1, size - 1)
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
        text(canvas, line, x, y + 12 + i * DEBUG_LINE_H, scale=0.38, color=AMBER)
    return y + DEBUG_LINE_H * len(lines) + 14


def draw_filters(canvas, activations, x, y, width):
    text(canvas, "Conv filters", x, y, scale=0.42, color=GREY)
    y += 10

    rows, cols = 4, 8
    pane_h = int(width * rows / cols)
    if activations is None:
        cv2.rectangle(canvas, (x, y), (x + width, y + pane_h), DIM, 1)
        return y + pane_h + 20

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

    resized = cv2.resize(mosaic, (width, pane_h), interpolation=cv2.INTER_NEAREST)
    canvas[y:y + pane_h, x:x + width] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return y + pane_h + 20


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
        color = DIM if dense is None else (value, value, 0)
        cv2.rectangle(canvas, (cx, cy), (cx + cell, cy + cell), color, -1)

    return y + rows * cell + (rows - 1) * gap + 20


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

    shown = state.text[-40:] if len(state.text) > 40 else state.text
    text(canvas, shown or "-", pad, top + 62, scale=0.8, color=WHITE, thickness=2)

    text(canvas, "q quit   space   backspace   c clear   r repeat", pad,
         layout.height - 12, scale=0.4, color=GREY)


TOP3_HEIGHT = 8 + 3 * 26 + 14


def compose(state, layout):
    """Build the single frame that goes to the one window."""
    canvas = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    canvas[:layout.camera_h, :layout.camera_w] = fit_into(
        state.camera, layout.camera_w, layout.camera_h)

    rail_x = layout.camera_w + layout.pad
    rail_w = layout.rail_w - 2 * layout.pad
    y = layout.pad + 10

    # The skeleton takes whatever the rail has left once the panels below are accounted
    # for. Sizing it from the remainder is what stops the motion readout from running
    # off the bottom of the column when --debug-motion is on.
    reserved = TOP3_HEIGHT + debug_height(state.debug)
    size = max(110, min(rail_w, layout.camera_h - y - layout.pad - reserved - 10))

    y = draw_skeleton(canvas, state, rail_x, y, rail_w, size)
    y = draw_top3(canvas, state, rail_x, y, rail_w)
    draw_debug(canvas, state.debug, rail_x, y)

    if layout.show_vis:
        vis_x = layout.camera_w + layout.rail_w + layout.pad
        vis_w = layout.vis_w - 2 * layout.pad
        vis_y = draw_filters(canvas, state.conv, vis_x, layout.pad + 10, vis_w)
        draw_neurons(canvas, state.dense, vis_x, vis_y, vis_w)

    cv2.line(canvas, (layout.camera_w, 0), (layout.camera_w, layout.camera_h), DIM, 1)
    if layout.show_vis:
        split = layout.camera_w + layout.rail_w
        cv2.line(canvas, (split, 0), (split, layout.camera_h), DIM, 1)

    draw_bottom(canvas, state, layout)
    return canvas
