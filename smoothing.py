"""Adaptive smoothing for the live hand landmarks.

MediaPipe's landmarks jitter by a pixel or two every frame even on a hand held
perfectly still, and that jitter is visible: the skeleton shimmers, and the classifier
sees a slightly different picture each frame. The obvious fix -- a fixed-alpha
exponential moving average -- has one dial, and that dial trades the two things we
want against each other. Enough smoothing to settle a still hand (the previous
`LANDMARK_ALPHA = 0.35`) also drags visibly behind a fast one.

The 1-euro filter (Casiez, Roussel & Vogel, CHI 2012) exists for exactly this
trade-off. Instead of one fixed alpha it recomputes the cutoff frequency every frame
from how fast the signal is currently moving:

    cutoff = MIN_CUTOFF + BETA * |speed|

Slow movement -> low cutoff -> heavy smoothing, which is what kills jitter, because
detection noise on a still hand *is* slow movement. Fast movement -> high cutoff ->
light smoothing, so a real gesture is not held back. Both ends improve at once, which
is the thing a single alpha cannot do.

The cutoff is computed per landmark, from that landmark's own speed, so a gesture
where one finger moves and the rest of the hand stays put -- a J, for instance -- gets
responsive tracking on the pinky without giving up the stillness of the palm.

This is inference-only and deliberately not in `preprocessing.py`: training data has
no temporal dimension, and that module has to stay importable from the training
environment with nothing but OpenCV and NumPy.
"""

import time

import numpy as np

# --- Tunables ------------------------------------------------------------------
# Cutoff floor, in Hz: what the filter falls back to when the hand is motionless.
# Lower means a stiller hand and more lag on slow drifts. Chosen so a still hand
# settles harder than the old fixed alpha managed (see `measure_smoothing.py`).
MIN_CUTOFF = 0.8

# How sharply the cutoff opens up with speed, in Hz per (pixel/second). This is the
# dial the whole thing turns on: 0 degenerates to a plain low-pass with a fixed
# alpha, and too high stops filtering anything that moves at all.
BETA = 0.02

# Cutoff for the speed estimate itself. 1.0 Hz is the value the paper recommends and
# there is little reason to move it -- it only smooths the derivative used to pick
# the cutoff, not the landmarks that get rendered.
D_CUTOFF = 1.0

# Clock guards. A frame gap outside this range means something stalled (a blocked
# window, a laptop resuming from sleep) rather than a real frame interval; dt is
# clamped so the filter neither divides by zero nor takes a huge step from a stale
# sample. Note a large dt legitimately drives alpha toward 1, i.e. "trust the new
# reading" -- which is the right call after a long gap anyway.
MIN_DT = 1e-3
MAX_DT = 1.0
# -------------------------------------------------------------------------------


def _alpha(cutoff, dt):
    """Smoothing factor for a first-order low-pass at `cutoff` Hz sampled every `dt`."""
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """1-euro filter over an array of points, vectorised across landmarks.

    Call it with the raw (21, 2) pixel landmarks each frame; it returns the filtered
    array. Call `reset()` whenever the hand is lost, so a hand reappearing somewhere
    else does not get dragged in from its old position.
    """

    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, x, timestamp=None):
        x = np.asarray(x, dtype=np.float32)
        if timestamp is None:
            timestamp = time.perf_counter()

        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = timestamp
            return x.copy()

        dt = float(np.clip(timestamp - self._t_prev, MIN_DT, MAX_DT))

        # Smooth the derivative before it decides the cutoff, otherwise a single noisy
        # frame would briefly unlock the filter and let that same noise straight through.
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Per-landmark speed magnitude, kept as (N, 1) so both coordinates of a point
        # share one cutoff. Using the magnitude rather than each axis separately keeps
        # diagonal movement from being filtered differently than straight movement.
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)

        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = timestamp
        return x_hat

    def effective_alpha(self, speed_px_s):
        """The alpha this filter would use at a given speed, for a 30 fps frame.

        Only used by the measurement script and for reasoning about the tunables --
        the filter itself never calls this.
        """
        cutoff = self.min_cutoff + self.beta * float(speed_px_s)
        return _alpha(cutoff, 1.0 / 30.0)
