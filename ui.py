"""Single-window interface for the live translator.

The translator used to open four OS windows — camera, skeleton, filter mosaic, neuron
grid — that had to be dragged into place every run, and which OpenCV decorates with a Qt
toolbar on Linux builds. Everything is composited into one frame here instead: one
window to place, and a layout that is ours rather than the window manager's.

Everything in this module works in **BGR**, because that is what `cv2.imshow` expects.
`preprocessing.render_skeleton` returns RGB, so the caller converts once on the way in.

The module is deliberately free of camera and model code so the whole interface can be
rendered from a synthetic state and inspected as a PNG, without a webcam.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

# --- Palette (BGR) -------------------------------------------------------------
BG = (27, 22, 20)           # #14161b
PANEL = (38, 31, 28)        # #1c1f26
EDGE = (58, 47, 42)         # #2a2f3a
TEXT = (238, 234, 232)      # #e8eaee
MUTED = (173, 161, 154)     # #9aa1ad
FAINT = (110, 97, 90)       # #5a616e
ACCENT = (127, 214, 70)     # #46d67f — confident
AMBER = (68, 181, 245)      # #f5b544 — in motion / partial
RED = (90, 85, 242)         # #f2555a — no hand
VIOLET = (250, 139, 167)    # #a78bfa — activations

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


@dataclass
class Layout:
    """Fixed geometry. Computed once, so the compositor never re-measures per frame."""

    camera_w: int = 720
    camera_h: int = 540
    rail_w: int = 300
    vis_w: int = 320
    bottom_h: int = 132
    pad: int = 14
    show_vis: bool = False

    @property
    def width(self):
        return self.camera_w + self.rail_w + (self.vis_w if self.show_vis else 0)

    @property
    def height(self):
        return self.camera_h + self.bottom_h


@dataclass
class ViewState:
    """Everything the interface needs for one frame."""

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
    fps: float = 0.0


# --- Primitives ----------------------------------------------------------------

def rounded_rect(img, x, y, w, h, color, radius=10):
    """Filled rounded rectangle. cv2 has no primitive for this."""
    radius = max(0, min(radius, w // 2, h // 2))
    if radius == 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        return
    cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), color, -1)
    for cx, cy in ((x + radius, y + radius), (x + w - radius, y + radius),
                   (x + radius, y + h - radius), (x + w - radius, y + h - radius)):
        cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)


# OpenCV's Hershey fonts are ASCII-only: anything else is drawn as a literal '?'. The
# prose in this project uses em dashes and ellipses freely, so rather than relying on
# every caller remembering, every string is folded down at the one place text is drawn.
_FALLBACKS = str.maketrans({
    "—": "-", "–": "-", "…": "...", "·": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'", "×": "x",
})


def ascii_only(string):
    return str(string).translate(_FALLBACKS).encode("ascii", "replace").decode("ascii")


def label(img, string, x, y, scale=0.44, color=MUTED, thickness=1, font=FONT):
    cv2.putText(img, ascii_only(string), (x, y), font, scale, color, thickness, cv2.LINE_AA)


def section_title(img, string, x, y):
    """Small caps-style header. cv2 has no letter-spacing, so spaces stand in."""
    label(img, " ".join(string.upper()), x, y, scale=0.36, color=FAINT, thickness=1)


def fit_into(image, width, height):
    """Scale preserving aspect ratio and letterbox onto a panel-coloured field."""
    canvas = np.full((height, width, 3), PANEL, dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas

    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x, y = (width - new_w) // 2, (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def normalize_to_gray(values):
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    return ((values - low) / (high - low) * 255).astype(np.uint8)


# --- Panels --------------------------------------------------------------------

def draw_camera(canvas, state, layout):
    """The webcam feed, with a status chip in the corner."""
    x0, y0 = 0, 0
    pane = fit_into(state.camera, layout.camera_w, layout.camera_h)
    canvas[y0:y0 + layout.camera_h, x0:x0 + layout.camera_w] = pane

    if state.hand_present:
        chip_color, chip_text = (AMBER, "MOVING") if state.moving else (ACCENT, "TRACKING")
    else:
        chip_color, chip_text = RED, "NO HAND"

    pad = layout.pad
    chip_w = 22 + 8 * len(chip_text)
    rounded_rect(canvas, pad, pad, chip_w, 26, PANEL, radius=13)
    cv2.circle(canvas, (pad + 14, pad + 13), 4, chip_color, -1, cv2.LINE_AA)
    label(canvas, chip_text, pad + 25, pad + 17, scale=0.4, color=TEXT)

    if state.fps:
        fps_text = f"{state.fps:.0f} fps"
        fw = 16 + 8 * len(fps_text)
        fx = layout.camera_w - pad - fw
        rounded_rect(canvas, fx, pad, fw, 26, PANEL, radius=13)
        label(canvas, fps_text, fx + 8, pad + 17, scale=0.4, color=MUTED)


def draw_skeleton_card(canvas, state, x, y, width, size=None):
    """'What the model sees' — the render that actually reaches the network."""
    section_title(canvas, "what the model sees", x, y)
    y += 12

    size = width if size is None else size
    rounded_rect(canvas, x, y, size, size, PANEL, radius=10)

    if state.skeleton is not None:
        inner = fit_into(state.skeleton, size - 8, size - 8)
        canvas[y + 4:y + 4 + size - 8, x + 4:x + 4 + size - 8] = inner
    else:
        label(canvas, "waiting for a hand", x + 18, y + size // 2, scale=0.42, color=FAINT)

    return y + size + 18


def draw_top3(canvas, state, x, y, width):
    """The three classes the model is weighing, with the commit threshold marked."""
    section_title(canvas, "top 3", x, y)
    y += 16

    if state.scores is None or not len(state.class_names):
        label(canvas, "-", x, y + 14, scale=0.5, color=FAINT)
        return y + 90

    order = np.argsort(state.scores)[::-1][:3]
    bar_x = x + 26
    bar_w = width - 26

    for rank, index in enumerate(order):
        probability = float(state.scores[index])
        row_y = y + rank * 30

        color = ACCENT if rank == 0 and probability > 0.5 else FAINT
        label(canvas, state.class_names[index].upper(), x, row_y + 15,
              scale=0.6, color=TEXT if rank == 0 else MUTED, thickness=1, font=FONT_BOLD)

        rounded_rect(canvas, bar_x, row_y + 4, bar_w, 14, PANEL, radius=7)
        filled = int(bar_w * probability)
        if filled > 3:
            rounded_rect(canvas, bar_x, row_y + 4, filled, 14, color, radius=7)

        percent = f"{probability * 100:.0f}%"
        label(canvas, percent, bar_x + bar_w - 10 * len(percent), row_y + 15,
              scale=0.4, color=TEXT if rank == 0 else MUTED)

    return y + 3 * 30 + 12


DEBUG_LINE_H = 15


def debug_height(lines):
    """Height `draw_debug` will occupy, so the rail can be budgeted before drawing."""
    return 0 if not lines else 14 + DEBUG_LINE_H * len(lines) + 12 + 14


def draw_debug(canvas, lines, x, y, width):
    """Live J/Z trajectory readings, for tuning the thresholds in motion.py."""
    if not lines:
        return y
    section_title(canvas, "motion", x, y)
    y += 14
    height = DEBUG_LINE_H * len(lines) + 12
    rounded_rect(canvas, x, y, width, height, PANEL, radius=8)
    for i, line in enumerate(lines):
        label(canvas, line, x + 10, y + 18 + i * DEBUG_LINE_H, scale=0.35, color=VIOLET)
    return y + height + 14


def draw_filter_mosaic(canvas, activations, x, y, width):
    """First convolutional block, tiled 4x8 — one cell per filter."""
    section_title(canvas, "conv filters", x, y)
    y += 12

    rows, cols = 4, 8
    if activations is None:
        rounded_rect(canvas, x, y, width, width // 2, PANEL, radius=10)
        return y + width // 2 + 18

    height, cell_w, filters = activations.shape[0], activations.shape[1], activations.shape[2]
    mosaic = np.zeros((height * rows, cell_w * cols), dtype=np.uint8)

    for i in range(min(filters, rows * cols)):
        feature = activations[:, :, i].copy()
        # Convolution padding lights up the border regardless of input; blanking it stops
        # the per-cell normalisation from being dominated by an artefact.
        feature[:2, :] = feature[-2:, :] = 0
        feature[:, :2] = feature[:, -2:] = 0
        r, c = divmod(i, cols)
        mosaic[r * height:(r + 1) * height, c * cell_w:(c + 1) * cell_w] = normalize_to_gray(feature)

    tinted = cv2.applyColorMap(mosaic, cv2.COLORMAP_BONE)
    pane_h = int(width * rows / cols)
    canvas[y:y + pane_h, x:x + width] = cv2.resize(tinted, (width, pane_h),
                                                   interpolation=cv2.INTER_NEAREST)
    return y + pane_h + 18


def draw_neuron_grid(canvas, dense, x, y, width):
    """The 128-unit hidden layer, one square per neuron."""
    section_title(canvas, "hidden layer  128", x, y)
    y += 12

    cols, rows = 16, 8
    gap = 2
    cell = (width - (cols - 1) * gap) // cols
    grid_h = rows * cell + (rows - 1) * gap

    rounded_rect(canvas, x - 6, y - 6, width + 12, grid_h + 12, PANEL, radius=8)

    intensities = normalize_to_gray(dense) if dense is not None else np.zeros(128, np.uint8)
    for i in range(128):
        r, c = divmod(i, cols)
        cx, cy = x + c * (cell + gap), y + r * (cell + gap)
        if dense is None:
            color = EDGE
        else:
            value = int(intensities[i])
            color = (int(VIOLET[0] * value / 255), int(VIOLET[1] * value / 255),
                     int(VIOLET[2] * value / 255))
        cv2.rectangle(canvas, (cx, cy), (cx + cell, cy + cell), color, -1)

    return y + grid_h + 18


def draw_bottom_bar(canvas, state, layout):
    """Current reading, commit progress, and the spelled-out text."""
    top = layout.camera_h
    width = layout.width
    pad = layout.pad

    cv2.rectangle(canvas, (0, top), (width, layout.height), BG, -1)
    cv2.line(canvas, (0, top), (width, top), EDGE, 1)

    # --- Left: the letter being read, and how close it is to committing.
    box_w = 190
    rounded_rect(canvas, pad, top + pad, box_w, layout.bottom_h - 2 * pad, PANEL, radius=12)

    if not state.hand_present:
        label(canvas, "-", pad + 30, top + 66, scale=1.9, color=FAINT, thickness=3, font=FONT_BOLD)
        label(canvas, "no hand", pad + 22, top + 92, scale=0.4, color=FAINT)
    elif state.uncertain:
        label(canvas, "?", pad + 28, top + 68, scale=1.6, color=AMBER, thickness=2, font=FONT_BOLD)
        label(canvas, "not sure", pad + 22, top + 92, scale=0.4, color=MUTED)
    else:
        label(canvas, state.letter.upper(), pad + 24, top + 72,
              scale=1.9, color=TEXT, thickness=3, font=FONT_BOLD)
        label(canvas, f"{state.confidence * 100:.0f}%", pad + 92, top + 60,
              scale=0.62, color=ACCENT, thickness=1, font=FONT_BOLD)

        # Progress toward COMMIT_FRAMES. Amber while filling, green the moment it lands.
        track_x, track_y, track_w = pad + 92, top + 74, box_w - 92 - 18
        rounded_rect(canvas, track_x, track_y, track_w, 8, EDGE, radius=4)
        filled = int(track_w * min(max(state.progress, 0.0), 1.0))
        if filled > 2:
            colour = ACCENT if state.progress >= 1.0 else AMBER
            rounded_rect(canvas, track_x, track_y, filled, 8, colour, radius=4)
        label(canvas, "hold to commit", track_x, top + 96, scale=0.34, color=FAINT)

    # --- Right: the buffer, which is the actual output of the whole program.
    text_x = pad * 2 + box_w
    section_title(canvas, "spelled", text_x, top + 26)

    shown = state.text[-32:] if len(state.text) > 32 else state.text
    if shown:
        label(canvas, shown, text_x, top + 68, scale=0.95, color=TEXT, thickness=2, font=FONT_BOLD)
        # A caret makes it read as a text field rather than a static label.
        (tw, _), _ = cv2.getTextSize(shown, FONT_BOLD, 0.95, 2)
        cv2.rectangle(canvas, (text_x + tw + 6, top + 48), (text_x + tw + 9, top + 70), ACCENT, -1)
    else:
        label(canvas, "start signing...", text_x, top + 68, scale=0.7, color=FAINT, font=FONT_BOLD)

    hints = "Q quit    SPACE space    BACKSPACE delete    C clear    R repeat letter"
    label(canvas, hints, text_x, layout.height - 16, scale=0.38, color=FAINT)


# --- Entry point ---------------------------------------------------------------

TOP3_HEIGHT = 16 + 3 * 30 + 12


def compose(state, layout):
    """Build the single frame that goes to the one window."""
    canvas = np.full((layout.height, layout.width, 3), BG, dtype=np.uint8)

    draw_camera(canvas, state, layout)

    rail_x = layout.camera_w + layout.pad
    rail_w = layout.rail_w - 2 * layout.pad
    y = layout.pad + 14

    # The skeleton card takes whatever the rail has left once the panels below it are
    # accounted for. Sizing it last-but-drawn-first is what stops the motion readout
    # from running off the bottom of the column when --debug-motion is on.
    reserved = TOP3_HEIGHT + debug_height(state.debug)
    available = layout.camera_h - y - layout.pad - reserved
    skeleton_size = max(120, min(rail_w, available - 12))

    y = draw_skeleton_card(canvas, state, rail_x, y, rail_w, size=skeleton_size)
    y = draw_top3(canvas, state, rail_x, y, rail_w)
    if state.debug:
        draw_debug(canvas, state.debug, rail_x, y, rail_w)

    if layout.show_vis:
        vis_x = layout.camera_w + layout.rail_w + layout.pad
        vis_w = layout.vis_w - 2 * layout.pad
        vis_y = layout.pad + 14
        vis_y = draw_filter_mosaic(canvas, state.conv, vis_x, vis_y, vis_w)
        draw_neuron_grid(canvas, state.dense, vis_x, vis_y + 6, vis_w)

    # Column separators, drawn last so no panel paints over them.
    cv2.line(canvas, (layout.camera_w, 0), (layout.camera_w, layout.camera_h), EDGE, 1)
    if layout.show_vis:
        split = layout.camera_w + layout.rail_w
        cv2.line(canvas, (split, 0), (split, layout.camera_h), EDGE, 1)

    draw_bottom_bar(canvas, state, layout)
    return canvas
