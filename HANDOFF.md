# Report: the retraining session, start to finish

> Written 2026-08-25, at the end of the Linux session that the previous handoff was asking
> for. The previous document was a to-do list; this one is a record of what happened when
> that list was executed, including the parts that did not go as the list predicted.
>
> **Audience:** whoever picks this up next, human or model. It is also the raw material for
> the release notes — the sections marked *for the README* are the ones worth lifting into
> public docs when this version ships.

---

## 1. What this session was for

The previous session rebuilt the training pipeline on Windows but could not run it: no GPU,
no datasets. Everything had been validated end-to-end against **synthetic** landmarks only.
The pipeline's mechanics were known to work; the model's quality was entirely unknown.

This session ran it for real. Outcome, up front:

| | |
|---|---|
| Validation accuracy | **96.8%** |
| Test accuracy (unseen dataset) | **93.7%** |
| Gap | 3.0 points |
| ONNX parity | 1.14e-05 |
| Live spelling | works — letters land first try |
| J / Z | work, after retuning the motion window |

The old model claimed 97.9% and flickered on a live webcam. The new one claims less and
behaves better, which is the entire point: the old number came from a leaky split.

---

## 2. Setup, and where the previous handoff was wrong

The five-step checklist in the old document was mostly right. Two things it did not
anticipate, both worth knowing because they cost real time:

### Python 3.12 is not installable from Arch repos

The old doc said "if not 3.11/3.12, install 3.12 aparte" without saying how. On CachyOS the
system Python is 3.14, there is no `python312` package, and neither TensorFlow nor MediaPipe
ships wheels for 3.13+. Resolved with `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12 .venv-infer
uv venv --python 3.12 .venv-train
```

### TensorFlow could not see the GPU, and the driver was fine

`nvidia-smi` worked, `tensorflow[and-cuda]` was installed, and
`tf.config.list_physical_devices('GPU')` still returned `[]`. The cause: those CUDA and cuDNN
wheels drop their shared objects under `site-packages/nvidia/*/lib`, and **TensorFlow does not
add that path to the loader itself.**

Fixed by replacing `.venv-train/bin/python` with a wrapper that builds `LD_LIBRARY_PATH`
before exec-ing the real interpreter, and mirroring it in `activate`:

```bash
#!/bin/bash
NVLIBS=$(find "$(dirname "$0")/../lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH}"
exec -a "$0" "/home/punpunia/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12" "$@"
```

Two dead ends before that one, recorded so nobody repeats them: a `sitecustomize.py` runs too
late (the loader path is read at process start, not at import), and a wrapper that exec's
`$0` recurses into itself. The wrapper must name the *real* interpreter by absolute path.

> **The venvs are not in git.** Anyone cloning fresh hits the same GPU problem and has to
> redo this. It is described in the README's *On GPUs* section.

### Everything else matched

Two separate environments really are mandatory (`mediapipe 0.10.21` pins `protobuf 4.25.9`,
TensorFlow 2.21 wants ≥5.28). The `<1.0` pin on MediaPipe is load-bearing: installing plain
`mediapipe` pulls 0.10.35, whose package contains only `modules/` and `tasks/` — no
`solutions`, so no `Hands` detector.

---

## 3. The datasets, and a decision the old plan did not contain

The old plan was: train on `alphabet`, test on `ayuraj`. That produces an honest number but a
weak model, because `alphabet` is ~78k images of **one signer**. Holding out the only other
signer means shipping a model that has never seen a second pair of hands.

So a third dataset was added — [`danrasband/asl-alphabet-test`](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test),
25 MB, a different person in different lighting, purpose-built as a companion test set for
`grassknoted/asl-alphabet`. That buys both halves at once: two signers in training, and a
test number that describes the model actually being shipped.

<p align="center">
  <img src="docs/dataset-split.svg" alt="Three photo datasets become one landmark file, split into train, validation and a held-out test set" width="100%">
</p>

**Detection rates** (`prepare_dataset.py` reports these; worth checking before every run):

| dataset | detected | rate |
|---|---|---|
| alphabet | 64,765 / 78,000 | 83.0% |
| ayuraj | 1,742 / 1,815 | 96.0% |
| danrasband | 765 / 780 | 98.1% |

`n` was the only class below 60% (56.6%). That matters later — see §6.

> **Trap, for whoever samples this data again:** a first pass with `--limit-per-class 50`
> reported **0% detection for `a` and `e`**, which looks alarming. It is an artefact. Files
> sort naturally, so the first 50 of a class are one contiguous burst from a single camera
> angle. The full run gave 74.5% and 77.2%. Do not tune anything on a small `--limit-per-class`
> sample.

Kaggle downloads need the CLI (`pip install kaggle` in `.venv-infer`) and an API token at
`~/.kaggle/access_token`.

---

## 4. Training

```bash
./.venv-train/bin/python train_model.py --landmarks landmarks.npz --epochs 40 --test-dataset danrasband
```

308,602 parameters. Ran 20 of 40 epochs, ~65 s each on the RTX 3050; `EarlyStopping` restored
epoch 12. Epoch 1 hit 78.6% training accuracy — worth noting because the failure mode the old
handoff warned about (`BN_MOMENTUM = 0.99` collapsing the model to a single class, validation
pinned at exactly 1/26 = 3.85%) would have been obvious right there. It did not happen.
`BN_MOMENTUM` stays at 0.9.

<p align="center">
  <img src="docs/training-accuracy.png" alt="Training and validation accuracy and loss over 20 epochs" width="640">
</p>

Training and validation track each other with a small, stable gap. No divergence, so no
memorisation — which is the thing this plot exists to show.

---

## 5. Two bugs found in code the previous session had written

Both were invisible against synthetic data and only surfaced against a real trained model.

### `export_onnx.py` failed its own parity check

Symptom: `Error: could not match output 'logits' (best error 0.060)`. Nothing was wrong with
the conversion. Two separate causes stacked:

1. **The probe input was uniform pixel noise** (`rng.uniform(0, 255, ...)`), which is nothing
   `render_skeleton()` could ever produce. So far outside BatchNorm's calibration that logits
   reached ±210, and at that magnitude ordinary float rounding exceeds the tolerance. Fixed by
   probing with an actual rendered skeleton. Error dropped 0.060 → 0.010 — still failing.
2. **Keras was running on the GPU, ONNX Runtime on the CPU.** The check was comparing cuDNN
   against ONNX Runtime — two kernel implementations — rather than measuring the conversion.
   Fixed by hiding the GPU inside `export_onnx.py`.

Final parity: **1.14e-05**, consistent with the 6.7e-06 the synthetic runs reported.

*Generalisable lesson: verify a converted model on the device it will actually run on, with
inputs from its real distribution. Both halves matter.*

### `evaluate.py` measured the wrong thing (not a bug, a gap)

Accuracy describes a bare `argmax`. The translator never uses a bare `argmax` — it commits a
letter only above `COMMIT_CONFIDENCE`. Two measurements were added:

- **Coverage and precision at the commit gate.** Test set: commits on 94.6% of hands, and is
  right 96.7% of the time when it does. That predicts felt reliability far better than 93.7%.
- **Jitter stability.** Re-render the same landmarks under small perturbations and count
  prediction flips. This measures the *original symptom* — a still hand whose prediction
  jumped — rather than a proxy for it.

Stability result, and the interesting part is the split:

| | agreement |
|---|---|
| Predictions that were correct | **99.0%** |
| Predictions that were wrong | 85.7% |

Correct predictions hold still; flicker is concentrated on genuinely ambiguous handshapes.
That is the shape you want. A model that flickered uniformly would be unstable; this one is
merely uncertain about the letters that *are* uncertain.

---

## 6. Limitations — read this before promising anything

**`n` is weak, and it compounds.** 54% recall on test, usually predicted as `m`. Two causes
multiply: M/N/S/T are all a fist with the thumb in a different place, *and* `n` had the worst
MediaPipe detection rate of any class (56.6%), so it reached training with the fewest samples.
Fixing it needs more `n` data specifically, not more epochs.

**The second signer is 2.6% of the training set.** `ayuraj` contributes 1,742 samples against
`alphabet`'s 64,765. `class_weight` in `train_model.py` balances *classes*, not *datasets*, so
the model is still dominated by one person's hands. The 3-point val/test gap is encouraging,
but do not read it as "subject-independent". More signers is the lever; epochs are not.

**J and Z are rules, not learning.** No ASL image dataset stores them as sequences, so
`motion.py` checks trajectory geometry by hand. In live testing J resolves well; **Z leans more
on its initial handshape than on the movement**, because `Z_MIN_REVERSALS = 2` demands two
clean direction changes. Acceptable, but it is pattern-matching, not recognition.

**Sign J and Z deliberately.** At 30 fps a fast gesture motion-blurs the hand enough for
MediaPipe to drop it, and `realtime_translator.py` calls `tracker.reset()` whenever the hand
disappears — so a rushed gesture destroys its own trajectory. `HISTORY_FRAMES` was raised
24 → 45 (0.8 s → 1.5 s) and `SMOOTHING` 3 → 5 for this.

**Do not lower `MOVING_THRESHOLD`.** This was tried, to catch slow gestures, and measured
before being kept — which is the only reason it did not ship. Instantaneous speed cannot
separate a slow gesture from a jittery still hand; the distributions overlap almost entirely:

| | still hand (light jitter) | still hand (heavy jitter) | slow J |
|---|---|---|---|
| median speed | 0.011 | 0.023 | 0.023 |

At `0.018`, a still hand holds the `COMMIT_FRAMES = 10` consecutive frames a letter needs only
**55%** of the time, and 0.8% under heavy jitter. Spelling breaks outright. It stays at 0.035,
and the reasoning is in a comment in `motion.py` so nobody re-derives it.

If someone does want to gate commits during slow gestures, **path length is the discriminator,
not speed** — still hand 0.16, slow J 0.90, slow Z 1.31 hand-widths, cleanly separated. It has
a cost: path stays elevated for the full window after moving between letters, so normal
spelling gets slower. It was offered and declined; spelling currently lands first try.

---

## 7. Visual assets — what exists and what each is for

Everything lives in `docs/`. All are current as of this session.

| File | What it shows | Regenerated by | For the README? |
|---|---|---|---|
| `dataset-split.svg` | Three sources → one npz → train/val/test, with the accuracies | hand-written | **yes** — the headline concept |
| `training-accuracy.png` | Accuracy and loss curves, 20 epochs | `train_model.py`, every run | **yes** |
| `confusion-test.png` | Confusion matrix, unseen dataset | `evaluate.py` | **yes** — the honest one |
| `confusion-validation.png` | Confusion matrix, validation | `evaluate.py` | optional |
| `render-check.png` | Contact sheet, one skeleton per letter | `check_render.py` | **yes** — shows what the net sees |
| `render-check-augment.png` | Same, augmented | `check_render.py --augment` | optional |
| `architecture.svg` | CNN layer diagram | hand-written | **yes** |
| `pipeline.svg` | Capture → detect → normalise → classify | hand-written | **yes** |

**Verified against the code this session:** `architecture.svg` matches `build_model()` exactly
(96×96×3 → 32/64/128 double-conv blocks → GAP → Dense 128 → Dense 26) and `pipeline.svg` is
correct (70% fill, 96×96, 26 letters). Neither needed changes.

`dataset-split.svg` is new — nothing illustrated the three-way split, which is this version's
main structural change.

Two caveats for whoever edits these:

- The hand-written SVGs are **light-themed with an explicit white background**. They stay
  readable on GitHub's dark theme, but they read as light cards. Making them theme-aware was
  deliberately not attempted — GitHub's SVG sanitiser makes `prefers-color-scheme` unreliable,
  and the `<picture>` two-file trick was more machinery than it was worth.
- Preview them with `rsvg-convert -w 1200 docs/x.svg -o /tmp/x.png` before committing. The
  first draft of `dataset-split.svg` had text overflowing the frame, invisible in source.

`generar_arquitectura.py`, `generar_infografia.py`, `arquitectura_3d.png` and
`infografia_educativa.png` are referenced by the old handoff but **do not exist in this
clone** — they were gitignored local assets from the Windows session. Nothing depends on them.

There is also an ASL alphabet reference sheet (one real photo per letter, built from the
training data) that was generated ad hoc for live testing and not kept. Rebuild it if needed —
it is genuinely useful when testing, since signing requires knowing the alphabet.

---

## 8. State of the repository

Committed nothing — the working tree is left for review.

```
 M README.md              metrics, three datasets, corrected licences, GPU notes
 M asl_cnn_model.onnx     retrained, 26 classes, replaces the 36-class model
 M class_names.txt        36 → 26
 M docs/training-accuracy.png
 M evaluate.py            +threshold metrics, +jitter stability
 M export_onnx.py         CPU-pinned parity, realistic probe input
 M motion.py              window 24 → 45, smoothing 3 → 5
 D asl_cnn_model.h5       deleted: 36-class legacy Keras, nothing referenced it
?? docs/dataset-split.svg
?? docs/confusion-test.png  docs/confusion-validation.png
?? docs/render-check.png    docs/render-check-augment.png
```

### Licence correction — do not skip this

The old README credited `MediaPipe_Processed_ASL_Dataset` under CC BY-SA 4.0. **That dataset
is no longer used.** The actual training data:

| Dataset | Role | Licence |
|---|---|---|
| `grassknoted/asl-alphabet` | main training set | **GPL-2.0** |
| `ayuraj/asl-dataset` | extra signers | CC0-1.0 |
| `danrasband/asl-alphabet-test` | held-out test | CC0-1.0 |

The dominant share of training data is **GPL-2.0**, not CC BY-SA. The README now states this
and points at the licences rather than interpreting them; whether weights are a derivative
work of training data is unsettled and this repo should not pretend otherwise.

---

## 9. Second chapter: the UI overhaul (Windows, 2026-08-26)

> Written after a separate Windows session took this branch from "retrained but still four
> separate OS windows" to "single window, ready for a public release." Same audience and
> purpose as §1 above — this picks up exactly where §8's `git status` snapshot left off.
> Unlike the retraining chapter, README.md was kept in sync commit-by-commit through all of
> this rather than updated once at the end, so it does not need a separate release-notes pass.

Outcome, up front: one window instead of four, a palette validated rather than picked by eye,
adaptive landmark smoothing that fixed a real motion-lag complaint, and four crash/UX bugs
caught and fixed from actual live testing on Windows — none of which showed up against the
synthetic-only testing the previous chapter had to rely on.

## 10. Single-window consolidation

The translator opened four separate OS windows — camera, skeleton, filter mosaic, neuron grid
— that had to be dragged into position on every run, and which OpenCV's Qt backend decorates
with a toolbar by default on Linux. `ui.py` composites all of it into one frame instead:
`ViewState` and `Layout` dataclasses plus a `compose()` function, with **no camera or model
code**, so the whole layout renders from a synthetic state and can be checked as a PNG without
a webcam — that is how it was built and how every bug below was caught before it ran live.

Layout: camera feed on the left; a rail with the skeleton view, top-3 bars, and (with
`--activations`) the hidden-layer grid; a third column, present only with `--activations`,
giving the filter mosaic the entire column height; a bottom bar with the current reading,
commit progress, the spelled-out word, and the two buttons (§11). `WINDOW_GUI_NORMAL` on
`cv2.namedWindow` suppresses the Qt toolbar.

Styling stays deliberately plain — this is a debugging surface for a model, not a product.

## 11. Two buttons: Filters (zoom) and Reference

Drawn regions hit-tested via `cv2.setMouseCallback`, since there is no widget toolkit here.

- **Filters (zoom)** pops the conv1 filter mosaic into its own full-size window. It works
  whether or not `--activations` is on: the model now always computes those activations when
  it has the named taps, regardless of the flag — only the *inline* column is gated by
  `--activations`. Previously the activations were only fetched under the flag at all, so
  peeking at them meant restarting.
- **Reference** opens `docs/asl-alphabet-reference.png` in a resizable window.
  `build_reference.py` generates that sheet: the primary path renders fresh through
  `preprocessing.render_skeleton()` from `landmarks.npz` when it's present (best quality, only
  works on a machine with the training data); the fallback slices and upscales the tiles
  already in `docs/render-check.png` when it isn't.

  **The image actually shipped is a real photo, not the generated skeleton sheet** — the user
  swapped in a WhatsApp-circulated ASL alphabet chart because it reads far more naturally as
  "how do I hold my hand" than the model's abstract landmark view does. Its provenance and
  licence are unknown. This was flagged at the time and not resolved — see §15.

Both buttons toggle closed on a second click and notice if the popup is closed with its own
window control instead (`realtime_translator.Popup`, §14).

## 12. Validated colour palette

Before: four unrelated saturated hues with no system behind them — neon green, neon amber, an
ad-hoc cyan for the hidden-layer heatmap, and a magenta hand-focus box left over from before
the single-window redesign that never got reconciled with it.

Replaced with three roles pulled from the dataviz skill's reference palette
(`references/palette.md`) rather than picked by eye, re-validated as a set with
`validate_palette.js` against a `#0d0d0d` surface — CVD separation, normal-vision separation
and contrast all pass:

- **GOOD** `#0ca30c` — a confident reading.
- **WARNING** `#fab219` — progress not yet committed.
- **INFO** `#3987e5` fading to `#0d366b` — one accent for "the AI is looking here," reused for
  both the hand-focus box and the hot end of the hidden-layer heatmap. One hue for one job,
  where there used to be two unrelated ones.

Ink hierarchy refined to the palette's warmer secondary/muted steps (not flat 128/128/128
grey); a `PANEL` surface (`#1a1a19`) now sits behind the skeleton/filter boxes, distinct from
the `BG` page plane (`#0d0d0d`, replacing pure black).

**Explicitly untouched:** the skeleton's finger colours, transcribed byte-for-byte from
MediaPipe's own drawing style so training and inference render the same picture. That palette
is correctness, not decoration — do not theme it.

## 13. Adaptive landmark smoothing — the 1-euro filter

Reported from live testing: fast hand movements looked visibly slowed down in the skeleton
view.

Cause: `LANDMARK_ALPHA = 0.35`, a fixed-alpha exponential moving average. Its lag in *time* is
constant — measured at 62 ms — so its lag in *pixels* grows linearly with speed: 19 px behind
at 300 px/s, 124 px behind at 2000 px/s.

Fix: `smoothing.py`'s `OneEuroFilter` recomputes the cutoff frequency every frame from the
landmark's own current speed instead of using one fixed alpha. Computed **per landmark**, so a
gesture that moves one finger (a J) tracks that finger responsively without giving up the
stillness of the rest of the hand.

Measured against the old filter (simulated 30 fps, 2 px detection noise):

| | fixed-alpha EMA | 1-euro |
|---|---|---|
| Still-hand shimmer (RMS) | 1.29 px | **1.00 px** |
| Lag at 300 px/s | 19 px / 62 ms | **5 px / 17 ms** |
| Lag at 2000 px/s | 124 px / 62 ms | **7 px / 4 ms** |

Effective alpha runs 0.14 at rest to 0.89 at 2000 px/s, crossing the old fixed 0.35 near
100 px/s.

**`motion.py`'s thresholds were re-checked against this, not assumed to still hold** — §6's
own warning about `MOVING_THRESHOLD` was measured through the *old* smoothing, so changing the
filter meant re-verifying rather than trusting it by inertia. Running the real `MotionTracker`
over synthetic still-hand and slow-J sequences: the still-vs-gesture speed separation flips
from **inverted** (0.74×, the exact overlap §6 documents) to **1.15×**, and still-hand path
length drops from 0.653 to 0.548 hand-widths — moving safely *below* `MIN_PATH_LENGTH = 0.55`
instead of sitting above it. `MOVING_THRESHOLD` and `MIN_PATH_LENGTH` themselves were **not**
changed; the margins around them widened instead.

Caveat, inherited from the commit and still true: those figures are synthetic. They show the
change moves things the right direction and breaks nothing measured — not that the thresholds
are now optimal against a real camera. See §15.

## 14. Four bugs found in live testing on Windows

All four were invisible against the synthetic checks `ui.py` was built against, and only
surfaced once the translator actually ran on the user's machine.

1. **Closing a popup crashed the whole translator.** `Popup.sync()` polls
   `WND_PROP_VISIBLE` to notice an OS-level close (`cv2.imshow` otherwise silently recreates a
   window the user just closed), but only caught `cv2.error`. On this Windows build, closing a
   popup with its native X apparently raises something else, which propagated straight out of
   the frame loop. Broadened to catch any exception in both `sync()` and `render()` — a
   misbehaving popup must never be able to kill the live camera loop.
2. **The main window closed and immediately reopened.** Same root cause, but the main window
   had no visibility check at all. The fix needed the check at the **top** of the frame loop,
   before that frame draws anything: checking *after* that frame's own `cv2.imshow` would only
   ever see the window `imshow` itself had just recreated — which is exactly the
   close-then-reopen symptom that was reported.
3. **The filter mosaic in the zoom popup looked stretched.** `large_filter_mosaic()` squeezed
   `FILTER_GRID`'s 8-rows-by-4-columns layout (twice as tall as wide) into a fixed square
   canvas, doubling every tile's width relative to its height. Fixed by sizing the popup from
   the grid's own aspect via a per-cell size instead of a fixed square; verified by rendering a
   circular test pattern and confirming it came out round, not oval.
4. **`iniciar_visualizador.bat` left the terminal open on a clean exit.** An unconditional
   `pause` at the end kept the window open even after pressing `q`. Changed to
   `if errorlevel 1 pause`, so only a real crash (non-zero exit code) leaves it open to read.

Also: the Reference window was made resizable (`WINDOW_NORMAL` instead of `WINDOW_AUTOSIZE`);
Filters stays a fixed size, since it is algorithmically generated rather than a photo someone
wants to zoom by hand. And on Windows specifically, title bars are themed dark via the DWM API
(`realtime_translator._dark_titlebar()`, `ctypes`) — `DWMWA_CAPTION_COLOR` for an exact black
match on Windows 11 (22000+), falling back to `DWMWA_USE_IMMERSIVE_DARK_MODE`'s dark grey on
Windows 10. A silent no-op on every other platform, since window decoration is the desktop
environment's job on Linux and there is no equivalent single API to reach for.

---

## 15. What is still open

From the retraining chapter, still genuinely unresolved:

1. **`n`.** The one weak class, needs more `n` images specifically, not more training.
2. **More signers.** The single highest-value improvement available, and the only real fix for
   the val/test gap.
3. **Z's motion detection**, if the rule-based version stops being good enough. A learned
   temporal model needs recorded sequences, which means capture tooling — explicitly declined
   twice now, so revisit that decision deliberately before building it.
4. **`SpellingBuffer` commits `i` before a slow J completes** in principle. Not observed live;
   the wider motion window from §6 may have made it moot. If it appears, §6 has the fix and its
   cost.

From the UI chapter:

5. **`motion.py`'s thresholds have only been re-verified against synthetic sequences**, after
   the §13 smoothing change. Never against a real camera. Worth a real pass if J/Z ever start
   feeling off, rather than assuming the synthetic re-check settled it for good.
6. **The shipped ASL alphabet reference photo has unknown provenance and licence** (§11).
   Resolve — confirm redistribution rights, or replace with the self-generated skeleton
   version `build_reference.py` already knows how to make — before treating it as a permanent
   public asset.
7. **The `legacy-ui` branch** (anchored at the commit right before the single-window redesign
   began — the retrained model, with the old four-window interface) exists for local testing
   preference. It is not actively developed in parallel with `main`; update it by hand if
   something from `main` is worth backporting. The §13 smoothing filter is a good candidate —
   it is not tied to the new UI and would fix the same "fast movement looks slow" symptom
   there too.

---

## 16. Decisions already made — do not re-litigate

Carried forward from both chapters. Do not re-open these without a specific reason:

- **26 classes, A–Z.** Digits dropped: too little data, and `0≡O`, `2≡V`, `6≡W`, `9≡F` are the
  same static handshape.
- **Reprocess from original photos**, never pre-rendered skeletons. `prepare_dataset.py` refuses
  the latter with an explicit error, because MediaPipe finds no hands in drawings.
- **No webcam capture tooling.** Consequence: J and Z must stay rule-based.
- **One renderer.** `preprocessing.render_skeleton()` is the only function that draws, and
  `preprocessing.py` imports nothing but `cv2` and `numpy` so both environments can call it.
  This is the invariant the entire redesign exists to protect — the original bug was training
  and inference drawing with different code. Do not add a second drawing path.
- **Named layers** `conv1_relu` and `dense_features` are the visualiser's contract, exposed as
  stable ONNX outputs. Renaming them breaks `--activations`.
- **`motion.py` takes raw pixel landmarks**, never normalised ones — normalisation re-centres
  the hand every frame, which is exactly what erases the movement J and Z depend on.
- **Single window is the public interface, shipped on `main`.** The four-separate-window
  layout lives on the `legacy-ui` branch for whoever prefers that for local testing/debugging.
  It is preserved, not deleted, and not being kept in active parallel development.
- **Landmark smoothing is adaptive (1-euro filter), not fixed-alpha.** Tunables live at the top
  of `smoothing.py`. Do not reintroduce a single fixed alpha — it cannot solve for a still hand
  and a fast gesture at the same time, which is the entire reason §13 exists.
- **UI colour is assigned by role, validated as a set, not picked per call.** Every colour in
  `ui.py` traces to the dataviz skill's reference palette (§12). Adding a new colour means
  validating it the same way, not eyeballing a hex value.
