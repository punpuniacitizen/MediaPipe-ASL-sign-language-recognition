import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import os
from mediapipe.framework.formats import landmark_pb2

def normalize_hand_landmarks(hand_landmarks):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x
    height = max_y - min_y

    center_x = min_x + width / 2
    center_y = min_y + height / 2

    # Hand fills ~70% of the canvas, matching the training preprocessing
    box_size = max(width, height) / 0.7
    if box_size == 0:
        box_size = 1.0

    normalized_landmarks = landmark_pb2.NormalizedLandmarkList()

    for lm in hand_landmarks.landmark:
        new_lm = normalized_landmarks.landmark.add()
        new_lm.x = (lm.x - (center_x - box_size / 2)) / box_size
        new_lm.y = (lm.y - (center_y - box_size / 2)) / box_size
        new_lm.z = lm.z

    return normalized_landmarks

def main():
    if not os.path.exists('class_names.txt'):
        print("Error: 'class_names.txt' not found. Run this script from the project root.")
        return

    with open('class_names.txt', 'r') as f:
        class_names = f.read().split(',')

    if not os.path.exists('asl_cnn_model.onnx'):
        print("Error: 'asl_cnn_model.onnx' not found. Run this script from the project root.")
        return

    print("Loading the ONNX model (starts instantly)...")
    import onnx
    try:
        onnx_model = onnx.load('asl_cnn_model.onnx')

        # Expose the first convolutional layer's output and the dense layer's
        # output, so the activation mosaic and neuron grid can read them.
        activation_tensor_name = 'sequential_1/conv2d_1/Relu:0'
        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = activation_tensor_name
        onnx_model.graph.output.append(intermediate_layer_value_info)

        dense_tensor_name = 'sequential_1/dense_1/Relu:0'
        dense_layer_value_info = onnx.helper.ValueInfoProto()
        dense_layer_value_info.name = dense_tensor_name
        onnx_model.graph.output.append(dense_layer_value_info)

        ort_sess = ort.InferenceSession(onnx_model.SerializeToString())
    except Exception as e:
        print(f"Error loading the ONNX model: {e}")
        return

    # Must match the canvas size used during training
    img_height = 64
    img_width = 64

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not access the webcam.")
        return

    print("\n--- STARTING TRANSLATOR ---")
    print("Place your hand in front of the camera. Press 'q' to quit.")

    input_name = ort_sess.get_inputs()[0].name

    # Rolling average over the last N predictions, so the displayed letter
    # doesn't flicker between classes frame to frame.
    history_len = 8
    score_history = []

    # Exponential moving average on the hand landmarks, so the skeleton
    # doesn't jitter with MediaPipe's per-frame noise.
    previous_landmarks = None
    alpha = 0.35  # lower = smoother/slower, higher = more responsive

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("No frame received from the camera. Skipping...")
                continue

            # Mirror the feed for a more natural view
            image = cv2.flip(image, 1)

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image.flags.writeable = True

            h, w, c = image.shape

            black_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            # Normalized 400x400 canvas fed to the model
            ia_canvas = np.zeros((400, 400, 3), dtype=np.uint8)
            prediction_label = "Looking for a hand..."

            if results.multi_hand_landmarks:
                # Only the first detected hand is processed, for stability
                hand_landmarks = results.multi_hand_landmarks[0]

                if previous_landmarks is not None:
                    for i in range(21):
                        curr_lm = hand_landmarks.landmark[i]
                        prev_lm = previous_landmarks[i]

                        curr_lm.x = alpha * curr_lm.x + (1 - alpha) * prev_lm[0]
                        curr_lm.y = alpha * curr_lm.y + (1 - alpha) * prev_lm[1]
                        curr_lm.z = alpha * curr_lm.z + (1 - alpha) * prev_lm[2]

                previous_landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

                normalized_hand = normalize_hand_landmarks(hand_landmarks)

                mp_drawing.draw_landmarks(
                    ia_canvas,
                    normalized_hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                mp_drawing.draw_landmarks(
                    black_canvas,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Bounding box on the raw feed, matching the crop the model sees
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                width = max_x - min_x
                height = max_y - min_y
                center_x = min_x + width / 2
                center_y = min_y + height / 2
                box_size = max(width, height) / 0.7

                x1_box = int((center_x - box_size / 2) * w)
                y1_box = int((center_y - box_size / 2) * h)
                x2_box = int((center_x + box_size / 2) * w)
                y2_box = int((center_y + box_size / 2) * h)

                cv2.rectangle(image, (x1_box, y1_box), (x2_box, y2_box), (255, 0, 255), 2)
                cv2.putText(image, "AI FOCUS", (x1_box + 5, y1_box + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

                resized_canvas = cv2.resize(ia_canvas, (img_width, img_height))

                # Keras keeps the layers.Rescaling(1./255) step inside the model,
                # so the ONNX graph expects raw float32 pixels in the 0-255 range.
                input_data = np.expand_dims(resized_canvas.astype(np.float32), axis=0)

                # outputs: [0] final logits, [1] conv layer activations, [2] dense layer (128 neurons)
                outputs = ort_sess.run(None, {input_name: input_data})
                predictions = outputs[0][0]

                activations = outputs[1][0]
                act_h, act_w, num_filters = activations.shape

                grid_h, grid_w = 4, 8
                mosaic = np.zeros((act_h * grid_h, act_w * grid_w), dtype=np.uint8)

                for i in range(min(num_filters, grid_h * grid_w)):
                    row = i // grid_w
                    col = i % grid_w
                    feature_map = activations[:, :, i].copy()
                    # Strip the convolution's padding artifacts at the edges
                    feature_map[0:2, :] = 0
                    feature_map[-2:, :] = 0
                    feature_map[:, 0:2] = 0
                    feature_map[:, -2:] = 0

                    f_min, f_max = feature_map.min(), feature_map.max()
                    if f_max > f_min:
                        normalized = ((feature_map - f_min) / (f_max - f_min) * 255).astype(np.uint8)
                    else:
                        normalized = np.zeros_like(feature_map, dtype=np.uint8)

                    y_start = row * act_h
                    y_end = y_start + act_h
                    x_start = col * act_w
                    x_end = x_start + act_w
                    mosaic[y_start:y_end, x_start:x_end] = normalized

                mosaic_resized = cv2.resize(mosaic, (mosaic.shape[1] * 2, mosaic.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('AI Filters', mosaic_resized)

                dense_activations = outputs[2][0]

                # Softmax computed manually: the ONNX graph outputs raw logits
                exp_preds = np.exp(predictions - np.max(predictions))
                score = exp_preds / exp_preds.sum()

                score_history.append(score)
                if len(score_history) > history_len:
                    score_history.pop(0)

                avg_scores = np.mean(score_history, axis=0)
                predicted_class = class_names[np.argmax(avg_scores)]
                confidence = 100 * np.max(avg_scores)

                if confidence > 50:
                    prediction_label = f"Sign: {predicted_class.upper()} ({confidence:.1f}%)"
                else:
                    prediction_label = "Not sure..."

                decision_canvas = np.zeros((400, 300, 3), dtype=np.uint8)

                cv2.putText(decision_canvas, "Hidden Layer (128 Neurons)", (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                d_min, d_max = dense_activations.min(), dense_activations.max()
                d_span = d_max - d_min if d_max > d_min else 1.0

                cell_size = 12
                gap = 3
                grid_start_x = 30
                grid_start_y = 40

                for i in range(128):
                    row = i // 16
                    col = i % 16
                    val = dense_activations[i]
                    norm_val = int((val - d_min) / d_span * 255)
                    color = (norm_val, norm_val, 0)

                    x1 = grid_start_x + col * (cell_size + gap)
                    y1 = grid_start_y + row * (cell_size + gap)
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size

                    cv2.rectangle(decision_canvas, (x1, y1), (x2, y2), color, -1)
                    cv2.rectangle(decision_canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)

                cv2.putText(decision_canvas, "AI Decision (Top 3)", (15, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                top_indices = np.argsort(avg_scores)[::-1][:3]

                bar_y_start = 210
                bar_height = 20
                bar_gap = 40

                for rank, idx in enumerate(top_indices):
                    prob = avg_scores[idx]
                    label = class_names[idx].upper()

                    text = f"{label}: {prob*100:.1f}%"
                    y_text = bar_y_start + rank * bar_gap + 15
                    cv2.putText(decision_canvas, text, (15, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    bar_w = int(prob * 120)
                    x_bar_start = 130
                    y_bar_start = bar_y_start + rank * bar_gap
                    x_bar_end = x_bar_start + bar_w
                    y_bar_end = y_bar_start + bar_height

                    # Green for a confident top guess, gray otherwise
                    bar_color = (0, 255, 0) if rank == 0 and prob > 0.5 else (128, 128, 128)
                    cv2.rectangle(decision_canvas, (x_bar_start, y_bar_start), (x_bar_end, y_bar_end), bar_color, -1)
                    cv2.rectangle(decision_canvas, (x_bar_start, y_bar_start), (x_bar_start + 120, y_bar_end), (255, 255, 255), 1)

                cv2.imshow('Decision Neurons', decision_canvas)
            else:
                # Clear the smoothing state so a lost hand doesn't bias the next one
                score_history.clear()
                previous_landmarks = None
                prediction_label = "Looking for a hand..."

                decision_canvas = np.zeros((400, 300, 3), dtype=np.uint8)
                cv2.putText(decision_canvas, "Hidden Layer (128 Neurons)", (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)

                cell_size = 12
                gap = 3
                grid_start_x = 30
                grid_start_y = 40

                for i in range(128):
                    row = i // 16
                    col = i % 16
                    x1 = grid_start_x + col * (cell_size + gap)
                    y1 = grid_start_y + row * (cell_size + gap)
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size
                    cv2.rectangle(decision_canvas, (x1, y1), (x2, y2), (20, 20, 20), -1)
                    cv2.rectangle(decision_canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)

                cv2.putText(decision_canvas, "Waiting for input...", (15, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
                cv2.imshow('Decision Neurons', decision_canvas)

            cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(image, prediction_label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('ASL Translator', image)
            cv2.imshow('AI View (Skeleton)', ia_canvas)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
