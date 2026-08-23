import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import os
from mediapipe.framework.formats import landmark_pb2

def normalize_hand_landmarks(hand_landmarks):
    # Obtener los límites (min/max) de los puntos de la mano
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Centro de la mano
    center_x = min_x + width / 2
    center_y = min_y + height / 2
    
    # Tamaño de la caja de la mano (con un margen de seguridad)
    # Ajustar para que la mano ocupe aproximadamente el 70% del canvas (0.7)
    box_size = max(width, height) / 0.7
    if box_size == 0:
        box_size = 1.0
        
    normalized_landmarks = landmark_pb2.NormalizedLandmarkList()
    
    for lm in hand_landmarks.landmark:
        new_lm = normalized_landmarks.landmark.add()
        # Mapear las coordenadas para centrar la mano y escalarla al tamaño fijo
        new_lm.x = (lm.x - (center_x - box_size / 2)) / box_size
        new_lm.y = (lm.y - (center_y - box_size / 2)) / box_size
        new_lm.z = lm.z
        
    return normalized_landmarks

def main():
    # 1. Cargar las clases (A, B, C, ..., 1, 2, 3...)
    if not os.path.exists('class_names.txt'):
        print("Error: No se encontró 'class_names.txt'. Debes ejecutar train_model.py primero.")
        return
    
    with open('class_names.txt', 'r') as f:
        class_names = f.read().split(',')

    # 2. Cargar el cerebro convertido a ONNX (Mucho más ligero que TensorFlow)
    if not os.path.exists('asl_cnn_model.onnx'):
        print("Error: No se encontró 'asl_cnn_model.onnx'. Espera a que termine la conversión.")
        return
        
    print("Cargando la IA en formato ONNX (arranca instantáneamente)...")
    import onnx
    try:
        # Cargamos el modelo original
        onnx_model = onnx.load('asl_cnn_model.onnx')
        # Agregamos la salida de la primera capa Convolucional
        activation_tensor_name = 'sequential_1/conv2d_1/Relu:0'
        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = activation_tensor_name
        onnx_model.graph.output.append(intermediate_layer_value_info)
        
        # Agregamos la salida de la primera capa densa (128 neuronas)
        dense_tensor_name = 'sequential_1/dense_1/Relu:0'
        dense_layer_value_info = onnx.helper.ValueInfoProto()
        dense_layer_value_info.name = dense_tensor_name
        onnx_model.graph.output.append(dense_layer_value_info)
        
        # Cargamos el modelo optimizado para inferencia en CPU
        ort_sess = ort.InferenceSession(onnx_model.SerializeToString())
    except Exception as e:
        print(f"Error al cargar el modelo ONNX: {e}")
        return

    # Mismo tamaño de lienzo que usamos en el entrenamiento
    img_height = 64
    img_width = 64

    # 3. Inicializar MediaPipe para encontrar la mano
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # 4. Iniciar la cámara (0 es normalmente la webcam integrada)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return
    
    print("\n--- INICIANDO TRADUCTOR ---")
    print("Coloca tu mano frente a la cámara. Presiona la tecla 'q' para salir.")

    # Obtenemos el nombre del nodo de entrada que espera el modelo ONNX
    input_name = ort_sess.get_inputs()[0].name

    # Historial para suavizar las predicciones (evita parpadeos o stutters)
    history_len = 8  # Número de frames para el promedio móvil
    score_history = []

    # Historial para suavizar el esqueleto de MediaPipe (evita temblores en los puntos)
    previous_landmarks = None
    alpha = 0.35  # Factor de suavizado (menor = más suave/lento, mayor = más rápido)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("No se recibió imagen de la cámara. Ignorando...")
                continue

            # Voltear la imagen horizontalmente para un efecto de espejo más natural
            image = cv2.flip(image, 1)

            # Convertir colores para MediaPipe
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image.flags.writeable = True

            # Obtener las dimensiones reales de la cámara
            h, w, c = image.shape
            
            # Creamos un lienzo negro del tamaño de la imagen original
            black_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            # Lienzo normalizado de 400x400 para la IA
            ia_canvas = np.zeros((400, 400, 3), dtype=np.uint8)
            prediction_label = "Buscando mano..."

            if results.multi_hand_landmarks:
                # Solo procesamos la primera mano detectada para mayor estabilidad
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Aplicar suavizado exponencial (EMA) a los puntos de la mano
                if previous_landmarks is not None:
                    for i in range(21):
                        curr_lm = hand_landmarks.landmark[i]
                        prev_lm = previous_landmarks[i]
                        
                        curr_lm.x = alpha * curr_lm.x + (1 - alpha) * prev_lm[0]
                        curr_lm.y = alpha * curr_lm.y + (1 - alpha) * prev_lm[1]
                        curr_lm.z = alpha * curr_lm.z + (1 - alpha) * prev_lm[2]
                
                # Guardamos los landmarks suavizados actuales para el siguiente frame
                previous_landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                
                # 1. Normalizar escala y posición para la IA (independiente de la distancia a la cámara)
                normalized_hand = normalize_hand_landmarks(hand_landmarks)
                
                # 2. Dibujar el esqueleto normalizado en el lienzo de la IA (400x400)
                mp_drawing.draw_landmarks(
                    ia_canvas,
                    normalized_hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # 3. Dibujar el esqueleto en el LIENZO NEGRO de visualización clásico
                mp_drawing.draw_landmarks(
                    black_canvas,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # 4. Dibujar el esqueleto en la IMAGEN REAL (para la webcam)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # 4.5 Calcular y dibujar la caja delimitadora (Focus Box) de la IA en la pantalla de la webcam
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                width = max_x - min_x
                height = max_y - min_y
                center_x = min_x + width / 2
                center_y = min_y + height / 2
                box_size = max(width, height) / 0.7
                
                # Coordenadas de la caja en píxeles de la cámara
                x1_box = int((center_x - box_size / 2) * w)
                y1_box = int((center_y - box_size / 2) * h)
                x2_box = int((center_x + box_size / 2) * w)
                y2_box = int((center_y + box_size / 2) * h)
                
                cv2.rectangle(image, (x1_box, y1_box), (x2_box, y2_box), (255, 0, 255), 2)
                cv2.putText(image, "ENFOQUE IA", (x1_box + 5, y1_box + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                
                # 5. Redimensionar el lienzo normalizado a 64x64 píxeles para la IA
                resized_canvas = cv2.resize(ia_canvas, (img_width, img_height))
                
                # Keras incluyó la capa de escalado layers.Rescaling(1./255) en el modelo,
                # por lo que el ONNX espera los píxeles originales en float32 (rango 0-255).
                input_data = np.expand_dims(resized_canvas.astype(np.float32), axis=0)
                
                # 4. ¡Hacer la predicción con ONNX Runtime!
                # outputs contiene: [0]=logits finales, [1]=capa conv, [2]=capa densa (128 neuronas)
                outputs = ort_sess.run(None, {input_name: input_data})
                predictions = outputs[0][0]
                
                # 5. Visualizar los mapas de activación de la capa convolucional
                activations = outputs[1][0]
                act_h, act_w, num_filters = activations.shape
                
                grid_h, grid_w = 4, 8
                mosaic = np.zeros((act_h * grid_h, act_w * grid_w), dtype=np.uint8)
                
                for i in range(min(num_filters, grid_h * grid_w)):
                    row = i // grid_w
                    col = i % grid_w
                    feature_map = activations[:, :, i].copy()
                    # Eliminar artefactos de borde (los extremos de la convolución por padding)
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
                
                # Escalar x2 para mejor visualización
                mosaic_resized = cv2.resize(mosaic, (mosaic.shape[1] * 2, mosaic.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('Filtros de la IA', mosaic_resized)
                
                # 6. Visualizar neuronas densas e inferencia
                dense_activations = outputs[2][0]  # Vector de 128 elementos
                
                # Calcular Softmax usando NumPy para obtener las probabilidades reales
                exp_preds = np.exp(predictions - np.max(predictions))
                score = exp_preds / exp_preds.sum()
                
                # Agregamos al historial para suavizar
                score_history.append(score)
                if len(score_history) > history_len:
                    score_history.pop(0)
                
                # Promediamos las probabilidades de los últimos N frames
                avg_scores = np.mean(score_history, axis=0)
                predicted_class = class_names[np.argmax(avg_scores)]
                confidence = 100 * np.max(avg_scores)
                
                # Solo mostrar si está moderadamente seguro
                if confidence > 50:
                    prediction_label = f"Senal: {predicted_class.upper()} ({confidence:.1f}%)"
                else:
                    prediction_label = "No estoy seguro..."
                
                # Crear el lienzo para mostrar el estado de las neuronas de la capa densa y decisiones
                decision_canvas = np.zeros((400, 300, 3), dtype=np.uint8)
                
                # Título de las neuronas
                cv2.putText(decision_canvas, "Capa Oculta (128 Neuronas)", (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Dibujar las 128 neuronas (grilla de 8x16)
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
                    
                    # Color cian/celeste para activaciones
                    color = (norm_val, norm_val, 0)
                    
                    x1 = grid_start_x + col * (cell_size + gap)
                    y1 = grid_start_y + row * (cell_size + gap)
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size
                    
                    cv2.rectangle(decision_canvas, (x1, y1), (x2, y2), color, -1)
                    cv2.rectangle(decision_canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)
                
                # Título de las decisiones
                cv2.putText(decision_canvas, "Decision de la IA (Top 3)", (15, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Obtener las top 3 probabilidades
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
                    
                    # Dibujar barra
                    bar_w = int(prob * 120)
                    x_bar_start = 130
                    y_bar_start = bar_y_start + rank * bar_gap
                    x_bar_end = x_bar_start + bar_w
                    y_bar_end = y_bar_start + bar_height
                    
                    # Verde para el ganador seguro, gris para los otros
                    bar_color = (0, 255, 0) if rank == 0 and prob > 0.5 else (128, 128, 128)
                    cv2.rectangle(decision_canvas, (x_bar_start, y_bar_start), (x_bar_end, y_bar_end), bar_color, -1)
                    cv2.rectangle(decision_canvas, (x_bar_start, y_bar_start), (x_bar_start + 120, y_bar_end), (255, 255, 255), 1)
                
                cv2.imshow('Neuronas de Decision', decision_canvas)
            else:
                # Si no hay mano, vaciamos el historial para evitar desfases
                score_history.clear()
                previous_landmarks = None
                prediction_label = "Buscando mano..."
                
                # Crear ventana de decisiones vacía
                decision_canvas = np.zeros((400, 300, 3), dtype=np.uint8)
                cv2.putText(decision_canvas, "Capa Oculta (128 Neuronas)", (15, 25),
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
                
                cv2.putText(decision_canvas, "Esperando datos...", (15, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
                cv2.imshow('Neuronas de Decision', decision_canvas)
            
            # Dibujar un recuadro de texto en la imagen principal
            cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(image, prediction_label, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Mostrar todo en pantalla
            cv2.imshow('Traductor ASL', image)
            cv2.imshow('Vision de la IA (Lienzo Negro)', ia_canvas)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
