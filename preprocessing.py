"""Single source of truth for turning a detected hand into model input.

Training, evaluation and live inference all go through `render_skeleton()`, so the
images the network sees on a webcam frame are identical in geometry, stroke weight and
channel order to the ones it was trained on. The previous version of this project
rendered the dataset with one piece of code and the live feed with another, which left
three silent mismatches:

  * the live canvas was an OpenCV BGR array fed straight to a model trained on RGB, so
    every finger arrived with its red and blue channels swapped;
  * the dataset hands sat at arbitrary positions and scales (29%-76% of the canvas)
    while the live hand was always centred at 70%;
  * the normalisation mixed width-relative and height-relative units, squashing the
    live skeleton to ~75% of its true width on a 4:3 camera.

Keeping the whole chain in one module is what stops those from reappearing.

This module deliberately imports nothing but OpenCV and NumPy. MediaPipe 0.10 pins
protobuf below 5 while current TensorFlow requires 5.28 or newer, so the two cannot
share an environment; keeping the renderer free of MediaPipe lets the training scripts
and the live translator both use it from whichever environment they run in. The palette
below is MediaPipe's own hand styling, transcribed so the drawing does not depend on an
API that version 1.0 removed.
"""

import cv2
import numpy as np

# Canvas the skeleton is drawn on before being resized down to the model input.
# Drawing above the target size and downsampling with INTER_AREA keeps the thin finger
# lines legible instead of aliasing them away. MediaPipe's stroke widths are specified
# for a 400 px canvas, so they are scaled from that reference; 192 renders 3.4x faster
# than 400 and differs by ~1% per pixel after the resize, which is far below anything
# the network can act on. Training and inference share this value, so whatever it is,
# both sides agree.
CANVAS_SIZE = 192
REFERENCE_CANVAS = 400

# Fraction of the canvas spanned by the hand's bounding box. Fixing this removes scale
# and position as variables: every hand reaches the network the same size.
HAND_FILL = 0.7

MODEL_INPUT_SIZE = 96

NUM_LANDMARKS = 21

# Shared by the offline dataset pass and the live translator. Keeping one value stops
# the training landmarks from being measured by a more accurate detector than the one
# running on the webcam.
HAND_MODEL_COMPLEXITY = 1

_WHITE_RGB = (224, 224, 224)

# MediaPipe's default hand styling, transcribed from
# mediapipe.python.solutions.drawing_styles as of 0.10.21 and converted from its BGR
# tuples to RGB. Values are (colour, circle_radius) and (colour, thickness), sized for
# a REFERENCE_CANVAS-wide canvas and scaled by `_scaled()`.
_RED, _PEACH, _PURPLE = (255, 48, 48), (255, 229, 180), (128, 64, 128)
_YELLOW, _GREEN, _BLUE, _GRAY = (255, 204, 0), (48, 255, 48), (21, 101, 192), (128, 128, 128)

_LANDMARK_STYLE = {
    0: (_RED, 5), 1: (_RED, 5), 5: (_RED, 5), 9: (_RED, 5), 13: (_RED, 5), 17: (_RED, 5),
    2: (_PEACH, 5), 3: (_PEACH, 5), 4: (_PEACH, 5),
    6: (_PURPLE, 5), 7: (_PURPLE, 5), 8: (_PURPLE, 5),
    10: (_YELLOW, 5), 11: (_YELLOW, 5), 12: (_YELLOW, 5),
    14: (_GREEN, 5), 15: (_GREEN, 5), 16: (_GREEN, 5),
    18: (_BLUE, 5), 19: (_BLUE, 5), 20: (_BLUE, 5),
}

_CONNECTION_STYLE = {
    (0, 1): (_GRAY, 3), (0, 5): (_GRAY, 3), (0, 17): (_GRAY, 3),
    (5, 9): (_GRAY, 3), (9, 13): (_GRAY, 3), (13, 17): (_GRAY, 3),
    (1, 2): (_PEACH, 2), (2, 3): (_PEACH, 2), (3, 4): (_PEACH, 2),
    (5, 6): (_PURPLE, 2), (6, 7): (_PURPLE, 2), (7, 8): (_PURPLE, 2),
    (9, 10): (_YELLOW, 2), (10, 11): (_YELLOW, 2), (11, 12): (_YELLOW, 2),
    (13, 14): (_GREEN, 2), (14, 15): (_GREEN, 2), (15, 16): (_GREEN, 2),
    (17, 18): (_BLUE, 2), (18, 19): (_BLUE, 2), (19, 20): (_BLUE, 2),
}

HAND_CONNECTIONS = frozenset(_CONNECTION_STYLE)


def _scaled(value, canvas_size, minimum=1):
    return max(minimum, int(round(value * canvas_size / REFERENCE_CANVAS)))


def landmarks_to_pixels(hand_landmarks, frame_width, frame_height):
    """MediaPipe result -> (21, 2) float32 array in real pixel units.

    MediaPipe normalises x by frame width and y by frame height. Converting to pixels
    first is what keeps the aspect ratio honest downstream.
    """
    return np.array(
        [(lm.x * frame_width, lm.y * frame_height) for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )


def normalize_landmarks(points_px):
    """Centre the hand and scale it so its bounding box spans HAND_FILL of the canvas.

    Takes and returns pixel-space points; the output is in [0, 1] canvas coordinates
    with the aspect ratio preserved, because a single `box_size` is applied to both
    axes and both axes are already in the same unit.
    """
    points_px = np.asarray(points_px, dtype=np.float32)

    lo = points_px.min(axis=0)
    hi = points_px.max(axis=0)
    extent = hi - lo

    box_size = float(extent.max()) / HAND_FILL
    if box_size <= 0:
        box_size = 1.0

    centre = lo + extent / 2.0
    origin = centre - box_size / 2.0

    return ((points_px - origin) / box_size).astype(np.float32)


def render_skeleton(norm_points, size=MODEL_INPUT_SIZE, canvas_size=CANVAS_SIZE):
    """Draw normalised landmarks as a skeleton. Returns (size, size, 3) uint8 **RGB**.

    The return value is genuinely RGB, not an OpenCV BGR buffer, so it can be handed to
    the model directly. Convert with `cv2.cvtColor(..., cv2.COLOR_RGB2BGR)` before
    showing it in an OpenCV window.
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    pts = np.rint(np.asarray(norm_points, dtype=np.float32) * canvas_size).astype(np.int32)

    for (start_idx, end_idx), (color, thickness) in _CONNECTION_STYLE.items():
        cv2.line(
            canvas,
            tuple(pts[start_idx]),
            tuple(pts[end_idx]),
            color,
            _scaled(thickness, canvas_size),
            cv2.LINE_AA,
        )

    # Points go on top of the lines, matching MediaPipe's own draw order.
    for idx, (color, radius) in _LANDMARK_STYLE.items():
        centre = tuple(pts[idx])
        scaled_radius = _scaled(radius, canvas_size)
        border_radius = max(scaled_radius + 1, int(scaled_radius * 1.2))
        cv2.circle(canvas, centre, border_radius, _WHITE_RGB, -1, cv2.LINE_AA)
        cv2.circle(canvas, centre, scaled_radius, color, -1, cv2.LINE_AA)

    if size != canvas_size:
        canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

    return canvas


def augment_landmarks(norm_points, rng, mirror=True):
    """Randomly perturb normalised landmarks.

    Augmenting in landmark space rather than pixel space keeps the stroke weight and
    the colour palette untouched, so every augmented sample still looks like something
    the renderer could legitimately produce at inference time.

    Mirroring is legitimate for ASL fingerspelling: a left-handed signer produces the
    mirror image of every letter, and the landmark indices keep their meaning, so the
    per-finger colours stay correct.
    """
    pts = np.asarray(norm_points, dtype=np.float32).copy()

    if mirror and rng.random() < 0.5:
        pts[:, 0] = 1.0 - pts[:, 0]

    centre = np.float32([0.5, 0.5])
    pts -= centre

    angle = np.deg2rad(rng.uniform(-15.0, 15.0))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.float32([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ rotation.T

    pts *= rng.uniform(0.9, 1.1)
    pts += centre
    pts += rng.uniform(-0.05, 0.05, size=2).astype(np.float32)
    pts += rng.normal(0.0, 0.005, size=pts.shape).astype(np.float32)

    # Keep every joint on the canvas, because cv2 would quietly clip a stray line and
    # the model would train on a truncated hand. Move the whole skeleton rather than
    # clamping individual joints, which would bend the fingers instead of shifting them.
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = float((hi - lo).max())
    if span > 1.0:
        pts = (pts - (lo + hi) / 2.0) / span + 0.5
        lo, hi = pts.min(axis=0), pts.max(axis=0)
    pts += np.clip(-lo, 0.0, None) - np.clip(hi - 1.0, 0.0, None)

    return pts.astype(np.float32)


def preprocess_hand(hand_landmarks, frame_width, frame_height, size=MODEL_INPUT_SIZE):
    """Full live-inference chain: MediaPipe result -> (normalised points, RGB image)."""
    points_px = landmarks_to_pixels(hand_landmarks, frame_width, frame_height)
    norm_points = normalize_landmarks(points_px)
    return norm_points, render_skeleton(norm_points, size=size)


def model_input(image_rgb):
    """Add the batch axis. The model keeps its own Rescaling layer, so pixels stay 0-255."""
    return np.expand_dims(image_rgb.astype(np.float32), axis=0)
