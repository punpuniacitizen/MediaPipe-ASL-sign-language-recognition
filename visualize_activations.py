import cv2
import mediapipe as mp
import numpy as np
import onnx
import onnxruntime as ort

print("\n--- INICIANDO VISUALIZADOR DE RED NEURONAL ---")
print("Cargando modelo y extrayendo conexiones internas...")

# 1. Cargar el modelo ONNX original
model_path = 'asl_cnn_model.onnx'
onnx_model = onnx.load(model_path)

# 2. Modificar el modelo en memoria para agregar la salida de la primera capa Convolucional
# Esto nos permite ver los "mapas de caracteristicas" que extrae la IA
activation_tensor_name = 'sequential_1/conv2d_1/Relu:0'
intermediate_layer_value_info = onnx.helper.ValueInfoProto()
intermediate_layer_value_info.name = activation_tensor_name
onnx_model.graph.output.append(intermediate_layer_value_info)

# 3. Iniciar la sesión de ONNX Runtime con el modelo modificado
ort_sess = ort.InferenceSession(onnx_model.SerializeToString())
input_name = ort_sess.get_inputs()[0].name

# 4. Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
img_width, img_height = 64, 64

print("¡Listo! Coloca tu mano frente a la cámara.")
print("Presiona 'q' en cualquier ventana para salir.")

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
            
            # Ejecutar el modelo modificado
            # outputs contiene ahora 2 elementos: [0] = prediccion final (softmax), [1] = activaciones de Conv2D
            outputs = ort_sess.run(None, {input_name: input_data})
            
            # Extraer las activaciones de la capa convolucional
            activations = outputs[1][0]  # Shape suele ser (62, 62, 32) o (64, 64, 32) dependiendo del padding
            act_h, act_w, num_filters = activations.shape
            
            # Crear un mosaico para visualizar los 32 filtros (4 filas x 8 columnas)
            grid_h, grid_w = 4, 8
            mosaic = np.zeros((act_h * grid_h, act_w * grid_w), dtype=np.uint8)
            
            for i in range(min(num_filters, grid_h * grid_w)):
                row = i // grid_w
                col = i % grid_w
                
                # Extraer el filtro individual
                feature_map = activations[:, :, i]
                
                # Normalizar los valores a 0-255 para poder verlos en pantalla como escala de grises
                f_min, f_max = feature_map.min(), feature_map.max()
                if f_max > f_min:
                    normalized = ((feature_map - f_min) / (f_max - f_min) * 255).astype(np.uint8)
                else:
                    normalized = np.zeros_like(feature_map, dtype=np.uint8)
                
                # Ubicarlo en el mosaico
                y_start = row * act_h
                y_end = y_start + act_h
                x_start = col * act_w
                x_end = x_start + act_w
                
                mosaic[y_start:y_end, x_start:x_end] = normalized
            
            # Escalar el mosaico (el doble de grande) para que sea visible
            mosaic_resized = cv2.resize(mosaic, (mosaic.shape[1] * 2, mosaic.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow('Mapas de Activacion (Filtros de la IA)', mosaic_resized)
            
        cv2.imshow('Camara Web', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
