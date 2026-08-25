"""Trajectory detection for J and Z, the two ASL letters that are movements.

The classifier works one frame at a time, so for J and Z it can only ever see the
handshape at the start or end of the gesture — the limitation the README documents.
This module watches the last second of hand positions and decides whether the motion
that accompanies a J-ish or Z-ish handshape is actually there.

Two things worth knowing about the implementation:

**It deliberately does not use the normalised landmarks.** `preprocessing.normalize_landmarks`
re-centres the hand on every frame, which is exactly what makes the classifier immune to
where the hand sits — and exactly what erases the motion we need here. So the tracker
takes raw pixel landmarks and divides by the hand's own size, giving displacements in
"hand widths": still immune to how far the signer is from the camera, but with the
movement intact.

**It is a rule-based detector, not a learned model.** Training a temporal model needs
recorded sequences, and every ASL image dataset stores J and Z as single frames. The
thresholds below are the tunable part; run the translator with `--debug-motion` to see
the measured values live and adjust them to your camera and signing speed.
"""

from collections import deque

import numpy as np

WRIST = 0
INDEX_TIP = 8
PINKY_TIP = 20

# Handshapes the classifier is likely to report at the start or end of each gesture.
J_SHAPES = frozenset({"i", "j"})
Z_SHAPES = frozenset({"d", "z"})

# --- Tunables ------------------------------------------------------------------
# Sized for a signer moving deliberately rather than quickly. At 30 fps a fast gesture
# smears the hand across the frame badly enough that MediaPipe drops it, and the
# translator clears this buffer whenever the hand disappears, so a rushed J loses its own
# trajectory before it can be measured. Signing slowly avoids that, but then the gesture
# outlasts a short window, hence 1.5 s rather than 0.8 s.
HISTORY_FRAMES = 45          # ~1.5 s at 30 fps
MIN_PATH_LENGTH = 0.55       # total travel, in hand widths, before anything counts
# Per-frame speed above which the hand counts as "in motion", which suppresses letter
# commits. Do not lower this to try to catch slow gestures: measured against MediaPipe
# jitter on a hand held still, the two distributions overlap almost completely (a still
# hand medians ~0.011 with light jitter and ~0.023 with heavy jitter, a slow J ~0.023).
# Dropping to 0.018 leaves a still hand only a ~55% chance of holding the COMMIT_FRAMES
# consecutive frames a letter needs, which breaks spelling outright. Instantaneous speed
# simply cannot separate a slow gesture from a jittery still hand; path length over the
# window can, and `stats()['path']` is the number to use for that.
MOVING_THRESHOLD = 0.035
J_MIN_DESCENT = 0.20         # J drops before it hooks
J_MIN_HOOK = 0.12            # horizontal travel of the hook at the end
Z_MIN_REVERSALS = 2          # a Z has two corners
Z_MIN_HORIZONTAL = 0.30      # the strokes are mostly sideways
# Raised with HISTORY_FRAMES to keep the same proportion of the window averaged; more
# frames of a slow gesture also means more idle wobble for `_reversals` to miscount.
SMOOTHING = 5                # frames averaged before measuring direction changes
# -------------------------------------------------------------------------------


def _reversals(values, deadzone):
    """Count sign changes in a sequence, ignoring wobble below `deadzone`."""
    sign = 0
    count = 0
    for value in values:
        if abs(value) < deadzone:
            continue
        current = 1 if value > 0 else -1
        if sign and current != sign:
            count += 1
        sign = current
    return count


class MotionTracker:
    """Rolling window of hand positions, measured in hand widths."""

    def __init__(self, history=HISTORY_FRAMES):
        self.buffer = deque(maxlen=history)

    def reset(self):
        self.buffer.clear()

    def update(self, points_px):
        """Feed raw pixel landmarks — not the normalised ones."""
        points_px = np.asarray(points_px, dtype=np.float32)
        extent = points_px.max(axis=0) - points_px.min(axis=0)
        hand_size = float(extent.max())
        if hand_size <= 0:
            return
        self.buffer.append(points_px / hand_size)

    def ready(self):
        return len(self.buffer) >= max(6, self.buffer.maxlen // 3)

    def _trajectory(self, index):
        track = np.stack([frame[index] for frame in self.buffer])
        if SMOOTHING > 1 and len(track) >= SMOOTHING:
            # Pad with the edge values so smoothing keeps the trajectory's length and
            # its endpoints. A plain 'valid' convolution eats the first frames, which
            # drags the anchor below into the middle of the first stroke and quietly
            # under-reports every displacement measured from it.
            pad = SMOOTHING // 2
            padded = np.pad(track, ((pad, pad), (0, 0)), mode="edge")
            kernel = np.ones(SMOOTHING) / SMOOTHING
            track = np.stack([np.convolve(padded[:, 0], kernel, "valid"),
                              np.convolve(padded[:, 1], kernel, "valid")], axis=1)[:len(track)]
        # Anchor to the start so the numbers are displacements, not absolute positions.
        return track - track[0]

    def speed(self):
        if len(self.buffer) < 2:
            return 0.0
        recent = np.stack([f[WRIST] for f in list(self.buffer)[-4:]])
        return float(np.linalg.norm(np.diff(recent, axis=0), axis=1).mean())

    def is_moving(self):
        return self.speed() > MOVING_THRESHOLD

    def stats(self, index):
        """Measurements for one fingertip's recent path."""
        if not self.ready():
            return None

        track = self._trajectory(index)
        steps = np.diff(track, axis=0)
        if len(steps) == 0:
            return None

        path_length = float(np.linalg.norm(steps, axis=1).sum())
        deadzone = max(path_length / 40.0, 1e-3)

        return {
            "path": path_length,
            "net": float(np.linalg.norm(track[-1])),
            # Image coordinates grow downward, so a positive dy is a descent.
            "descent": float(track[:, 1].max() - track[0, 1]),
            # Total width of the stroke, which is what a Z's zigzag makes large — and
            # stays meaningful even when the window opens mid-gesture.
            "horizontal": float(track[:, 0].max() - track[:, 0].min()),
            "hook": float(abs(track[-1, 0] - track[np.argmax(track[:, 1]), 0])),
            "reversals": _reversals(steps[:, 0], deadzone),
        }

    def classify(self, static_label):
        """Return 'j', 'z' or None. Only ever overrides a compatible handshape."""
        label = (static_label or "").lower()

        if label in J_SHAPES:
            s = self.stats(PINKY_TIP)
            if s and s["path"] >= MIN_PATH_LENGTH and s["descent"] >= J_MIN_DESCENT and s["hook"] >= J_MIN_HOOK:
                return "j"
            # An 'i' that never moved is a genuine 'i'.
            return "i" if label == "j" and s and s["path"] < MIN_PATH_LENGTH else None

        if label in Z_SHAPES:
            s = self.stats(INDEX_TIP)
            if (s and s["path"] >= MIN_PATH_LENGTH and s["reversals"] >= Z_MIN_REVERSALS
                    and s["horizontal"] >= Z_MIN_HORIZONTAL):
                return "z"
            return "d" if label == "z" and s and s["path"] < MIN_PATH_LENGTH else None

        return None

    def debug_lines(self, static_label):
        """Live readout of the measured values, for tuning the thresholds above."""
        label = (static_label or "").lower()
        index, target = (PINKY_TIP, "J") if label in J_SHAPES else (INDEX_TIP, "Z")
        s = self.stats(index)
        if not s:
            return [f"motion: filling buffer ({len(self.buffer)})"]
        return [
            f"target {target}   speed {self.speed():.3f}",
            f"path       {s['path']:.2f} / {MIN_PATH_LENGTH}",
            f"descent    {s['descent']:.2f} / {J_MIN_DESCENT}",
            f"hook       {s['hook']:.2f} / {J_MIN_HOOK}",
            f"horizontal {s['horizontal']:.2f} / {Z_MIN_HORIZONTAL}",
            f"reversals  {s['reversals']} / {Z_MIN_REVERSALS}",
        ]
