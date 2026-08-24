"""Live ASL fingerspelling translator.

Reads hand landmarks from the webcam, renders them through `preprocessing.render_skeleton()`
— the exact function the model was trained on — and classifies the result. Stable letters
accumulate into a word buffer, and the two dynamic letters are resolved by `motion.py`.

    python realtime_translator.py
    python realtime_translator.py --activations     # filter mosaic + neuron grid
    python realtime_translator.py --debug-motion    # live J/Z trajectory readout

Keys: q quit · space · backspace · c clear · r repeat last letter
"""

import argparse
import os

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

import preprocessing as pp
from motion import MotionTracker

# Outputs written by export_onnx.py. A model exported any other way still runs; the
# extra visualisation panels are simply unavailable.
LOGITS = "logits"
CONV_TAP = "conv1_relu"
DENSE_TAP = "dense_features"

CONFIDENCE_FLOOR = 0.50      # below this the reading is reported as uncertain
COMMIT_CONFIDENCE = 0.75     # and this much is needed to append a letter
COMMIT_FRAMES = 10           # consecutive agreeing frames before committing
SMOOTHING_WINDOW = 8         # rolling average over recent probability vectors
LANDMARK_ALPHA = 0.35        # exponential smoothing on the landmarks themselves


class SpellingBuffer:
    """Turns a stream of per-frame guesses into typed text.

    A letter is committed once it has been the top prediction for COMMIT_FRAMES frames
    in a row while the hand is still. The same letter will not be committed twice in a
    row — otherwise a hand held steady would spray repeats — so double letters need the
    'r' key, or a brief move away and back.
    """

    def __init__(self):
        self.text = ""
        self._candidate = None
        self._streak = 0
        self._last_committed = None

    def update(self, letter, confidence, hand_present, moving):
        if not hand_present:
            self._candidate = None
            self._streak = 0
            self._last_committed = None
            return None

        if moving:
            # Mid-gesture frames are meaningless for a static classifier.
            self._streak = 0
            return None

        if letter != self._candidate:
            self._candidate = letter
            self._streak = 1
            if letter != self._last_committed:
                self._last_committed = None
        else:
            self._streak += 1

        if (self._streak == COMMIT_FRAMES and confidence >= COMMIT_CONFIDENCE
                and letter != self._last_committed):
            self.text += letter.upper()
            self._last_committed = letter
            return letter
        return None

    def progress(self):
        return min(self._streak / COMMIT_FRAMES, 1.0)

    def repeat(self):
        if self.text and self.text[-1] != " ":
            self.text += self.text[-1]

    def space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "
        self._last_committed = None

    def backspace(self):
        self.text = self.text[:-1]
        self._last_committed = None

    def clear(self):
        self.text = ""
        self._last_committed = None


def load_session(path):
    if not os.path.exists(path):
        raise SystemExit(f"Error: '{path}' not found. Run this script from the project root.")

    session = ort.InferenceSession(path)
    names = [o.name for o in session.get_outputs()]
    input_meta = session.get_inputs()[0]
    size = input_meta.shape[1]
    if not isinstance(size, int):
        size = pp.MODEL_INPUT_SIZE

    taps = CONV_TAP in names and DENSE_TAP in names
    if not taps:
        print("Note: this model has no named activation taps, so the visualiser panels")
        print("      are disabled. Re-export with export_onnx.py to enable them.")

    return session, input_meta.name, size, names, taps


def draw_activation_mosaic(activations, grid=(4, 8), scale=3):
    height, width, filters = activations.shape
    rows, cols = grid
    mosaic = np.zeros((height * rows, width * cols), dtype=np.uint8)

    for i in range(min(filters, rows * cols)):
        feature = activations[:, :, i].copy()
        # Convolution padding lights up the border regardless of the input; blanking it
        # keeps the normalisation below from being dominated by an artefact.
        feature[:2, :] = feature[-2:, :] = 0
        feature[:, :2] = feature[:, -2:] = 0

        low, high = feature.min(), feature.max()
        cell = ((feature - low) / (high - low) * 255).astype(np.uint8) if high > low \
            else np.zeros_like(feature, dtype=np.uint8)

        r, c = divmod(i, cols)
        mosaic[r * height:(r + 1) * height, c * width:(c + 1) * width] = cell

    return cv2.resize(mosaic, (mosaic.shape[1] * scale, mosaic.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST)


def draw_neuron_panel(dense, scores, class_names, active):
    canvas = np.zeros((400, 300, 3), dtype=np.uint8)
    label_color = (255, 255, 255) if active else (128, 128, 128)
    cv2.putText(canvas, "Hidden Layer (128 Neurons)", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)

    cell, gap, x0, y0 = 12, 3, 30, 40
    if active and dense is not None:
        low, high = dense.min(), dense.max()
        span = high - low if high > low else 1.0

    for i in range(128):
        r, c = divmod(i, 16)
        x1, y1 = x0 + c * (cell + gap), y0 + r * (cell + gap)
        if active and dense is not None:
            value = int((dense[i] - low) / span * 255)
            color = (value, value, 0)
        else:
            color = (20, 20, 20)
        cv2.rectangle(canvas, (x1, y1), (x1 + cell, y1 + cell), color, -1)
        cv2.rectangle(canvas, (x1, y1), (x1 + cell, y1 + cell), (0, 0, 0), 1)

    if not active:
        cv2.putText(canvas, "Waiting for input...", (15, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
        return canvas

    cv2.putText(canvas, "AI Decision (Top 3)", (15, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for rank, idx in enumerate(np.argsort(scores)[::-1][:3]):
        probability = scores[idx]
        y = 210 + rank * 40
        cv2.putText(canvas, f"{class_names[idx].upper()}: {probability * 100:.1f}%", (15, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        color = (0, 255, 0) if rank == 0 and probability > 0.5 else (128, 128, 128)
        cv2.rectangle(canvas, (130, y), (130 + int(probability * 120), y + 20), color, -1)
        cv2.rectangle(canvas, (130, y), (250, y + 20), (255, 255, 255), 1)

    return canvas


def draw_hud(image, reading, buffer, progress, debug_lines):
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(image, reading, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Progress toward committing the current letter.
    if progress > 0:
        cv2.rectangle(image, (0, 58), (int(w * progress), 60), (0, 200, 255), -1)

    cv2.rectangle(image, (0, h - 70), (w, h), (0, 0, 0), -1)
    text = buffer.text[-28:] if len(buffer.text) > 28 else buffer.text
    cv2.putText(image, text or "-", (20, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "q quit  space  backspace  c clear  r repeat", (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    for i, line in enumerate(debug_lines):
        cv2.putText(image, line, (w - 250, 80 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="asl_cnn_model.onnx")
    parser.add_argument("--classes", default="class_names.txt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--activations", action="store_true", help="Show the filter mosaic and neuron grid")
    parser.add_argument("--debug-motion", action="store_true", help="Overlay the J/Z trajectory measurements")
    args = parser.parse_args()

    if not os.path.exists(args.classes):
        raise SystemExit(f"Error: '{args.classes}' not found. Run this script from the project root.")
    class_names = open(args.classes).read().strip().split(",")

    session, input_name, size, output_names, has_taps = load_session(args.model)
    print(f"Model: {args.model} | input {size}x{size} | {len(class_names)} classes")

    show_panels = args.activations and has_taps
    wanted = [LOGITS] if LOGITS in output_names else [output_names[0]]
    if show_panels:
        wanted += [CONV_TAP, DENSE_TAP]

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit("Error: could not access the webcam.")

    buffer = SpellingBuffer()
    tracker = MotionTracker()
    score_history = []
    previous = None

    mp_hands = mp.solutions.hands
    print("\n--- STARTING TRANSLATOR ---")
    print("Place your hand in front of the camera. Press 'q' to quit.")

    with mp_hands.Hands(model_complexity=pp.HAND_MODEL_COMPLEXITY,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:

        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            frame.flags.writeable = False
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.flags.writeable = True

            h, w = frame.shape[:2]
            skeleton_rgb = np.zeros((size, size, 3), dtype=np.uint8)
            reading = "Looking for a hand..."
            dense = None
            scores = None
            debug_lines = []

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]

                # Exponential smoothing on the raw landmarks, so the skeleton does not
                # jitter with MediaPipe's per-frame noise.
                points_px = pp.landmarks_to_pixels(landmarks, w, h)
                if previous is not None:
                    points_px = LANDMARK_ALPHA * points_px + (1 - LANDMARK_ALPHA) * previous
                previous = points_px

                tracker.update(points_px)
                normalized = pp.normalize_landmarks(points_px)
                skeleton_rgb = pp.render_skeleton(normalized, size=size)

                outputs = session.run(wanted, {input_name: pp.model_input(skeleton_rgb)})
                logits = outputs[0][0]
                if show_panels:
                    dense = outputs[2][0]

                # The graph emits raw logits; softmax happens here.
                exponentials = np.exp(logits - logits.max())
                score_history.append(exponentials / exponentials.sum())
                if len(score_history) > SMOOTHING_WINDOW:
                    score_history.pop(0)

                scores = np.mean(score_history, axis=0)
                letter = class_names[int(np.argmax(scores))]
                confidence = float(scores.max())

                resolved = tracker.classify(letter)
                if resolved and resolved in class_names:
                    letter = resolved

                moving = tracker.is_moving()
                buffer.update(letter, confidence, True, moving)

                if confidence > CONFIDENCE_FLOOR:
                    reading = f"Sign: {letter.upper()} ({confidence * 100:.1f}%)"
                    if moving:
                        reading += "  [moving]"
                else:
                    reading = "Not sure..."

                if args.debug_motion:
                    debug_lines = tracker.debug_lines(letter)

                # Draw the hand and the region the model actually sees.
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

                extent = points_px.max(axis=0) - points_px.min(axis=0)
                box = extent.max() / pp.HAND_FILL
                centre = points_px.min(axis=0) + extent / 2
                x1, y1 = (centre - box / 2).astype(int)
                x2, y2 = (centre + box / 2).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, "AI FOCUS", (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

                if show_panels:
                    cv2.imshow("AI Filters", draw_activation_mosaic(outputs[1][0]))
            else:
                score_history.clear()
                previous = None
                tracker.reset()
                buffer.update(None, 0.0, False, False)

            if show_panels:
                cv2.imshow("Decision Neurons",
                           draw_neuron_panel(dense, scores, class_names, scores is not None))

            draw_hud(frame, reading, buffer, buffer.progress(), debug_lines)
            cv2.imshow("ASL Translator", frame)
            # render_skeleton returns RGB; OpenCV windows expect BGR.
            cv2.imshow("AI View (Skeleton)", cv2.cvtColor(skeleton_rgb, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                buffer.space()
            elif key in (8, 127):
                buffer.backspace()
            elif key == ord("c"):
                buffer.clear()
            elif key == ord("r"):
                buffer.repeat()

    capture.release()
    cv2.destroyAllWindows()
    if buffer.text:
        print(f"\nTyped: {buffer.text}")


if __name__ == "__main__":
    main()
