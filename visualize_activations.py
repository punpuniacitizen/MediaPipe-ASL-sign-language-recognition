import cv2
import mediapipe as mp
import numpy as np
import onnx
import onnxruntime as ort

print("\n--- STARTING NEURAL NETWORK VISUALIZER ---")
print("Loading model and exposing its internal connections...")

model_path = 'asl_cnn_model.onnx'
onnx_model = onnx.load(model_path)

# Expose the first convolutional layer's output, so we can see the
# feature maps the network extracts from each frame.
activation_tensor_name = 'sequential_1/conv2d_1/Relu:0'
intermediate_layer_value_info = onnx.helper.ValueInfoProto()
intermediate_layer_value_info.name = activation_tensor_name
onnx_model.graph.output.append(intermediate_layer_value_info)

ort_sess = ort.InferenceSession(onnx_model.SerializeToString())
input_name = ort_sess.get_inputs()[0].name

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
img_width, img_height = 64, 64

print("Ready! Place your hand in front of the camera.")
print("Press 'q' in any window to quit.")

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image.flags.writeable = True

        h, w, c = image.shape
        black_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                black_canvas,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            crop_size = min(h, w)
            start_x = (w - crop_size) // 2
            start_y = (h - crop_size) // 2
            square_canvas = black_canvas[start_y:start_y+crop_size, start_x:start_x+crop_size]

            resized_canvas = cv2.resize(square_canvas, (img_width, img_height))
            input_data = np.expand_dims(resized_canvas.astype(np.float32), axis=0)

            # outputs: [0] final prediction (softmax), [1] Conv2D activations
            outputs = ort_sess.run(None, {input_name: input_data})

            # Shape is typically (62, 62, 32) or (64, 64, 32) depending on padding
            activations = outputs[1][0]
            act_h, act_w, num_filters = activations.shape

            grid_h, grid_w = 4, 8
            mosaic = np.zeros((act_h * grid_h, act_w * grid_w), dtype=np.uint8)

            for i in range(min(num_filters, grid_h * grid_w)):
                row = i // grid_w
                col = i % grid_w

                feature_map = activations[:, :, i]

                # Normalize to 0-255 grayscale for display
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

            cv2.imshow('Activation Maps (AI Filters)', mosaic_resized)

        cv2.imshow('Webcam', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
