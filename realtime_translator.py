"""Live ASL fingerspelling translator.

Reads hand landmarks from the webcam, renders them through `preprocessing.render_skeleton()`
— the exact function the model was trained on — and classifies the result. Stable letters
accumulate into a word buffer, and the two dynamic letters are resolved by `motion.py`.

    python realtime_translator.py
    python realtime_translator.py --activations     # filter mosaic + neuron grid inline
    python realtime_translator.py --debug-motion    # live J/Z trajectory readout

Everything is composited into a single window by `ui.py`, rather than the four separate
OS windows earlier versions opened. Two buttons in the bottom-right open secondary
windows on click: "Filters (zoom)" pops the convolutional filter mosaic full-size
regardless of --activations, and "Reference" opens a static ASL alphabet sheet (see
build_reference.py) to sign against. Both toggle closed on a second click, or by closing
the popup window directly.

Keys: q quit · space · backspace · c clear · r repeat last letter
"""

import argparse
import ctypes
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

import preprocessing as pp
import ui
from motion import MotionTracker

WINDOW = "ASL Translator"
FILTERS_WINDOW = "Conv Filters (enlarged)"
REFERENCE_WINDOW = "ASL Alphabet Reference"
REFERENCE_IMAGE = "docs/asl-alphabet-reference.png"  # see build_reference.py

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


def _dark_titlebar(window_name):
    """Best-effort: theme a cv2 window's native title bar dark, via the Windows DWM
    API, so it doesn't clash with the pure-black interior the way a stock white title
    bar does.

    Windows only, and a silent no-op everywhere else -- window decoration is the
    desktop environment's job on Linux, and each one themes it differently already,
    so there's no equivalent to reach for. DWMWA_CAPTION_COLOR needs Windows 11
    22000+; DWMWA_USE_IMMERSIVE_DARK_MODE alone (Windows 10 1809+) still gets a dark
    grey title bar instead of white, which is most of the improvement even where the
    exact-black match isn't available. Both calls fail silently (a non-zero HRESULT,
    not a Python exception) on unsupported versions, and everything here is wrapped
    besides, since a mismatched title bar is cosmetic and must never be able to break
    the translator the way an unguarded popup call once did.
    """
    if sys.platform != "win32":
        return
    try:
        user32 = ctypes.windll.user32
        dwmapi = ctypes.windll.dwmapi
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.FindWindowW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        dwmapi.DwmSetWindowAttribute.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]

        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return

        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        DWMWA_CAPTION_COLOR = 35
        dark = ctypes.c_int(1)
        dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                                     ctypes.byref(dark), ctypes.sizeof(dark))
        black = ctypes.c_uint(0x00000000)  # COLORREF 0x00BBGGRR, matches ui.py's BLACK
        dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_CAPTION_COLOR,
                                     ctypes.byref(black), ctypes.sizeof(black))

        # Without this the new attributes only take visual effect after the window is
        # next moved or resized; SWP_FRAMECHANGED forces the title bar to redraw now.
        SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER, SWP_FRAMECHANGED = 1, 2, 4, 0x20
        user32.SetWindowPos(hwnd, None, 0, 0, 0, 0,
                            SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED)
    except Exception:
        pass


class Popup:
    """A secondary cv2 window toggled by a button, that also notices if the user closes
    it with the window manager's own control instead of clicking the button again.

    OpenCV has no callback for a window closing, so the only way to detect that is to
    poll WND_PROP_VISIBLE. Without this, clicking the button to close a popup the user
    had already dismissed by hand would do nothing (imshow silently recreates whatever
    window it's given), and the button would keep reporting the popup as open forever.

    Every cv2 call here is wrapped defensively. How HighGUI reports -- or reacts to --
    a window the user just closed with their own window manager varies across
    backends: on at least one Windows build, closing a popup with its native X could
    take the whole translator down with it, not just that window, once the resulting
    cv2 call raised something the earlier narrower `except cv2.error` didn't catch.
    A popup misbehaving should never be able to kill the live camera loop.
    """

    def __init__(self, name, resizable=False):
        self.name = name
        self.resizable = resizable
        self.open = False
        self._created = False

    def toggle(self):
        self.open = not self.open

    def sync(self):
        """Call once per frame, before rendering. Detects an OS-level close."""
        if self._created and self.open:
            try:
                if cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                    self.open = False
            except Exception:
                self.open = False

    def render(self, image):
        try:
            if self.open:
                first_time = not self._created
                if first_time:
                    flags = cv2.WINDOW_GUI_NORMAL
                    flags |= cv2.WINDOW_NORMAL if self.resizable else cv2.WINDOW_AUTOSIZE
                    cv2.namedWindow(self.name, flags)
                cv2.imshow(self.name, image)
                self._created = True
                if first_time:
                    # After imshow, not namedWindow: the native window isn't guaranteed
                    # to be fully realised (and findable by title) until an image has
                    # actually been shown in it.
                    _dark_titlebar(self.name)
            elif self._created:
                cv2.destroyWindow(self.name)
                self._created = False
        except Exception:
            self.open = False
            self._created = False


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

    # Taps are fetched whenever the model has them, not only under --activations: the
    # zoom button needs fresh activations to pop up on demand even when the inline
    # column isn't showing. --activations only controls whether that inline column
    # (and the hidden-layer grid under it) render by default.
    show_panels = args.activations and has_taps
    wanted = [LOGITS] if LOGITS in output_names else [output_names[0]]
    if has_taps:
        wanted += [CONV_TAP, DENSE_TAP]

    reference_image = cv2.imread(REFERENCE_IMAGE) if os.path.exists(REFERENCE_IMAGE) else None
    if reference_image is None:
        print(f"Note: '{REFERENCE_IMAGE}' not found; run build_reference.py to enable "
              "the Reference button.")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit("Error: could not access the webcam.")

    buffer = SpellingBuffer()
    tracker = MotionTracker()
    layout = ui.Layout(show_vis=show_panels)
    score_history = []
    previous = None

    filters_popup = Popup(FILTERS_WINDOW)
    reference_popup = Popup(REFERENCE_WINDOW, resizable=True)
    mouse = {"hover": None}

    def on_mouse(event, x, y, _flags, _param):
        mouse["hover"] = ui.hit_test(layout, x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if mouse["hover"] == "filters" and has_taps:
                filters_popup.toggle()
            elif mouse["hover"] == "reference" and reference_image is not None:
                reference_popup.toggle()

    # WINDOW_GUI_NORMAL suppresses the toolbar and status bar that OpenCV's Qt backend
    # adds by default on Linux builds; the interface is composited here, not by Qt.
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    titlebar_done = {"main": False}  # applied once, after the first real imshow

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
            state = ui.ViewState(class_names=class_names, text=buffer.text)

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

                state.skeleton = cv2.cvtColor(skeleton_rgb, cv2.COLOR_RGB2BGR)
                state.scores = scores
                state.letter = letter
                state.confidence = confidence
                state.hand_present = True
                state.moving = moving
                state.uncertain = confidence <= CONFIDENCE_FLOOR
                state.text = buffer.text
                state.progress = buffer.progress()
                if has_taps:
                    state.conv = outputs[1][0]
                    state.dense = outputs[2][0]
                if args.debug_motion:
                    state.debug = tracker.debug_lines(letter)
            else:
                score_history.clear()
                previous = None
                tracker.reset()
                buffer.update(None, 0.0, False, False)
                state.text = buffer.text

            state.camera = frame
            state.filters_enabled = has_taps
            state.reference_enabled = reference_image is not None
            state.filters_open = filters_popup.open
            state.reference_open = reference_popup.open
            state.hover = mouse["hover"]
            cv2.imshow(WINDOW, ui.compose(state, layout))
            if not titlebar_done["main"]:
                _dark_titlebar(WINDOW)
                titlebar_done["main"] = True

            # render() runs every frame regardless of .open, since closing (whether by
            # button or by the popup's own window control) is handled inside it too --
            # only the relatively expensive mosaic resize is skipped while closed.
            filters_popup.sync()
            reference_popup.sync()
            filters_popup.render(ui.large_filter_mosaic(state.conv) if filters_popup.open else None)
            reference_popup.render(reference_image if reference_popup.open else None)

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
