# Traspaso: ProyectoNN (traductor ASL) — continuar en Linux

> Documento de estado para retomar el trabajo. Escrito el 2026-08-24 al final de una sesión en
> Windows donde se rediseñó el pipeline pero **no se llegó a reentrenar el modelo**, porque eso
> necesita GPU y los datasets originales. Todo lo que falta está acá.
>
> Se puede borrar cuando el reentrenamiento esté hecho y el README refleje los resultados reales.

---

## 1. Resumen en treinta segundos

El traductor ASL funcionaba pero las predicciones saltaban violentamente con la mano quieta. La
causa **no era falta de épocas**: eran tres bugs que hacían que el modelo viera en producción algo
distinto de lo que vio al entrenar, más una métrica de validación inflada por fuga de datos.

El pipeline se rehízo entero alrededor de una idea: **entrenamiento e inferencia dibujan el
esqueleto con la misma función**. Todo el código nuevo está escrito y probado end-to-end con datos
sintéticos. Falta correrlo con datos reales.

**Tu tarea:** montar el entorno en Linux, bajar los datasets de fotos, correr los cuatro pasos del
pipeline, y actualizar el README con las métricas reales.

---

## 2. Primeros cinco minutos en Linux

Si estás arrancando esta sesión recién entrando a Linux, este es el orden exacto. Cada paso depende
del anterior — no saltear ninguno, y si alguno falla, parar ahí y resolverlo antes de seguir.

```bash
# 1. La GPU tiene que estar visible ANTES de instalar nada. Si esto falla, todo lo
#    demás es tiempo perdido hasta que se resuelva el driver de NVIDIA.
nvidia-smi

# 2. Version de Python. Ni TensorFlow ni MediaPipe tienen wheels para 3.13+.
python3 --version   # necesita ser 3.11 o 3.12; si no, instalar 3.12 aparte

# 3. Clonar la branch de trabajo (NO main -- ahi todavia esta el codigo viejo).
#    Clonar a un disco local, nunca dentro de una carpeta sincronizada (el I/O lento
#    fue uno de los problemas originales del proyecto).
git clone --branch wip/retrain-pipeline \
    https://github.com/punpuniacitizen/MediaPipe-ASL-sign-language-recognition.git \
    ~/proyecto-nn
cd ~/proyecto-nn

# 4. Dos entornos separados -- mediapipe y TensorFlow fijan versiones de protobuf
#    incompatibles entre si (ver la seccion 6 para el detalle completo).
python3.12 -m venv .venv-infer
./.venv-infer/bin/pip install -r requirements.txt

python3.12 -m venv .venv-train
./.venv-train/bin/pip install -r requirements-train.txt

# 5. Confirmar que TensorFlow ve la GPU. Tiene que devolver un PhysicalDevice, no una
#    lista vacia. Si sale vacio, parar y diagnosticar antes de instalar nada mas.
./.venv-train/bin/python -c \
    "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Si los cinco pasos salen bien, seguir por la sección 8 (datasets) y 9 (el pipeline en sí). El resto
de este documento es contexto y referencia — no hace falta leerlo entero antes de empezar, pero
conviene tenerlo a mano para las secciones 5 (invariantes que no romper), 10 (verificación) y 11
(trampas ya resueltas, para no reintroducirlas).

---

## 3. Estado actual

| Cosa | Estado |
|---|---|
| Código del pipeline | ✅ Escrito y probado end-to-end (con datos sintéticos) |
| `asl_cnn_model.onnx` en el repo | ⚠️ **Todavía el viejo, 36 clases, entrenado con los bugs** |
| `class_names.txt` | ⚠️ **Todavía 36 clases**; `train_model.py` lo reescribe con 26 |
| `docs/training-accuracy.png` | ⚠️ **Curva vieja**; `train_model.py` la regenera |
| README | ✅ Actualizado, **sin afirmar ninguna métrica** (a propósito) |
| Entrenamiento real | ❌ **Pendiente** — es lo que hay que hacer |
| Umbrales de J/Z | ⚠️ Calibrados contra trayectorias sintéticas, no contra una mano real |

El traductor **corre hoy** con el modelo viejo: lee el tamaño de entrada y la cantidad de clases del
propio modelo, así que no rompe. Los paneles de visualización quedan desactivados porque el modelo
viejo no tiene las salidas nombradas.

---

## 4. El diagnóstico (por qué se rehizo todo)

Medido alimentando al modelo viejo con imágenes del propio dataset:

| entrada | accuracy |
|---|---|
| como entrenó: sin normalizar, RGB | **98.6%** |
| + centrada al 70% (lo que hacía el traductor) | **75.3%** |
| + canales R/B invertidos | **39.6%** |

### Bug 1 — canales R/B invertidos (costaba ~49 puntos por sí solo)

MediaPipe define su paleta como tuplas **BGR**. El dataset guardaba los colores visualmente
correctos, y `image_dataset_from_directory(color_mode='rgb')` los entregaba en orden **RGB**: canal 0
= R. Pero el traductor pasaba `ia_canvas`, un array de OpenCV en orden **BGR**, sin convertir: canal
0 = B. El meñique entrenaba con canal 0 = 21 y en vivo recibía 192. Cada dedo llegaba con el color
cambiado.

### Bug 2 — normalización asimétrica

Las manos del dataset ocupaban entre **29% y 76%** del lienzo en posiciones arbitrarias (medido sobre
imágenes reales). El traductor las centraba siempre al **70%**. El modelo nunca vio esa distribución.

### Bug 3 — deformación por mezcla de unidades

`box_size = max(width, height)` combinaba coordenadas normalizadas por el **ancho** del frame con
otras normalizadas por el **alto**, y después dibujaba sobre un lienzo cuadrado. En una cámara 4:3 el
esqueleto se achataba al **75%** de su ancho real.

### Bug 4 — el 97.9% de validación era mentira

Split aleatorio 80/20 sobre archivos que eran variantes casi idénticas del mismo frame (`A0.jpg`,
`A0 (2).jpg`, …, y `hand1_0_bot_seg_1..5`). Casi cada imagen de validación tenía un gemelo en
entrenamiento. **No medía generalización.**

---

## 5. Invariantes del diseño — NO ROMPER

Estas propiedades son la razón de ser del rediseño. Si alguna se rompe, vuelven los bugs.

1. **`preprocessing.render_skeleton()` es la única función que dibuja.** La llaman el pipeline de
   entrenamiento, `evaluate.py`, `check_render.py` y el traductor. Nunca dibujar un esqueleto en otro
   lado.
2. **`preprocessing.py` importa SOLO `cv2` y `numpy`.** No agregarle mediapipe. Ver §7: mediapipe y
   TensorFlow no pueden convivir, y este módulo tiene que ser importable desde ambos entornos. La
   paleta de mediapipe está transcrita adentro (verificada byte a byte contra la original).
3. **`render_skeleton()` devuelve RGB genuino**, no un buffer BGR de OpenCV. Para mostrarlo en una
   ventana de cv2 hay que convertir con `cv2.cvtColor(..., cv2.COLOR_RGB2BGR)`.
4. **`CANVAS_SIZE` y `HAND_FILL` son compartidos.** Cambiar cualquiera obliga a reentrenar.
5. **Las capas `conv1_relu` y `dense_features` tienen nombre explícito** en
   `train_model.build_model()`. `export_onnx.py` las expone como salidas ONNX con nombre estable. Si
   se les cambia el nombre, se rompe el visualizador.
6. **`motion.py` consume landmarks en píxeles crudos, NO normalizados.** La normalización recentra la
   mano en cada frame, que es justamente lo que borra el movimiento que J y Z necesitan.
7. **`BN_MOMENTUM = 0.9` en `train_model.py`.** No volver al 0.99 de Keras. Ver §11.

---

## 6. Inventario de archivos

### Pipeline (todo nuevo o reescrito)

| Archivo | Rol |
|---|---|
| `preprocessing.py` | **Núcleo.** Normalización, render compartido, aumentación. Solo cv2+numpy. |
| `prepare_dataset.py` | MediaPipe una vez sobre las fotos originales → `landmarks.npz` |
| `train_model.py` | Reescrito: split por bloques, aumentación en landmarks, BN+dropout+GAP |
| `evaluate.py` | Matriz de confusión, métricas por clase, dos accuracies separadas |
| `export_onnx.py` | **No existía.** Exporta con salidas nombradas + verifica paridad |
| `check_render.py` | Contact sheet de lo que ve la red. La verificación más importante |
| `motion.py` | Detección de J y Z por trayectoria |
| `realtime_translator.py` | Reescrito: deletreo, `--activations`, `--debug-motion` |

`visualize_activations.py` fue borrado; su función es ahora la bandera `--activations` del traductor.
Está en el historial de git si hiciera falta.

### Requirements — dos archivos, a propósito

- `requirements.txt` → inferencia y `prepare_dataset.py` (mediapipe, opencv, onnxruntime, numpy<2)
- `requirements-train.txt` → `train_model.py`, `evaluate.py`, `export_onnx.py` (tensorflow, tf2onnx,
  sklearn, matplotlib, opencv, numpy)

### Constantes que importan

```python
# preprocessing.py
CANVAS_SIZE = 192            # lienzo de dibujo; 3.4x más rápido que 400, ~1% de diferencia
REFERENCE_CANVAS = 400       # los grosores de mediapipe están definidos para este tamaño
HAND_FILL = 0.7              # fracción del lienzo que ocupa el bbox de la mano
MODEL_INPUT_SIZE = 96        # entrada del modelo (antes 64)
HAND_MODEL_COMPLEXITY = 1    # compartido: dataset e inferencia miden con el mismo detector

# train_model.py
BN_MOMENTUM = 0.9            # NO usar el 0.99 de Keras — ver §11

# realtime_translator.py
CONFIDENCE_FLOOR = 0.50      # debajo de esto muestra "Not sure..."
COMMIT_CONFIDENCE = 0.75     # confianza mínima para agregar una letra al buffer
COMMIT_FRAMES = 10           # frames seguidos de acuerdo antes de confirmar
SMOOTHING_WINDOW = 8         # promedio móvil de probabilidades
LANDMARK_ALPHA = 0.35        # suavizado exponencial de los landmarks

# motion.py  (calibrar contra una mano real — ver §10)
HISTORY_FRAMES = 24          # ~0.8 s a 30 fps
MIN_PATH_LENGTH = 0.55       # recorrido mínimo, en anchos de mano
MOVING_THRESHOLD = 0.035     # velocidad por encima de la cual la mano "se mueve"
J_MIN_DESCENT = 0.20
J_MIN_HOOK = 0.12
Z_MIN_REVERSALS = 2
Z_MIN_HORIZONTAL = 0.30
```

---

## 7. Setup en Linux

### Trampa central: mediapipe y TensorFlow NO conviven

```
mediapipe 0.10.21  requiere  protobuf<5,>=4.25.3   y  numpy<2
tensorflow 2.20    requiere  protobuf>=5.28
```

Son mutuamente incompatibles. **Hacen falta dos virtualenvs.** Por eso `preprocessing.py` no importa
mediapipe: así los dos entornos pueden renderizar igual.

### Segunda trampa: mediapipe 1.0 borró la API que usa el proyecto

mediapipe 1.0.x eliminó `mediapipe.solutions` por completo (solo quedan `modules` y `tasks`). No hay
reemplazo directo para el detector `Hands`. El pin `mediapipe>=0.10.21,<1.0` en `requirements.txt` es
**imprescindible**, no precautorio. Un `pip install mediapipe` pelado rompe el proyecto.

### Tercera trampa: la versión de Python

Ni TensorFlow ni MediaPipe publican wheels para Python 3.13+. **Usar 3.11 o 3.12.**

### Comandos

```bash
# Sacar el proyecto de OneDrive: el I/O sincronizado fue el cuello de botella original
cp -r "/ruta/al/ProyectoNN" ~/proyecto-nn && cd ~/proyecto-nn

python3.12 -m venv .venv-infer
./.venv-infer/bin/pip install -r requirements.txt

python3.12 -m venv .venv-train
./.venv-train/bin/pip install -r requirements-train.txt
```

### Verificar la GPU

```bash
./.venv-train/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Tiene que listar la RTX 3050. El extra `[and-cuda]` trae las wheels de CUDA y cuDNN, así que el
driver NVIDIA es la única dependencia de sistema (el 577.05 que ya está instalado sobra).

`train_model.py` ya llama a `set_memory_growth` para que TF no tome los 6 GB de una.

**Sobre precisión mixta:** existe la bandera `--mixed-precision` pero está **apagada por defecto** y
está bien así. El cuello de botella es el render en CPU, no la GPU, y complica la exportación a ONNX.

---

## 8. Los datasets

Hay que bajar las **fotos originales**, no esqueletos ya renderizados. El `archive/` que está en el
proyecto contiene esqueletos pre-renderizados: **no sirve** para este pipeline (`prepare_dataset.py`
falla con un mensaje explícito si se lo apunta ahí, porque MediaPipe no detecta manos en dibujos).

Por los patrones de nombre, el `archive/` actual mezcla al menos dos fuentes:
- letras con nombres tipo `A0.jpg` / `A1000.jpg` → estilo *ASL Alphabet* de Kaggle
- dígitos tipo `hand1_0_bot_seg_1_cropped` → estilo `ayuraj/asl-dataset`

Como el proyecto pasó a **26 letras**, hace falta:

1. **Un dataset principal de letras A–Z.** El *ASL Alphabet* de Kaggle (~3000 img/clase, 200×200,
   ~1 GB). Ojo: es de **un solo sujeto** en condiciones casi idénticas, así que por sí solo
   generaliza mal.
2. **Un segundo dataset de letras con otros sujetos**, por ejemplo las letras de `ayuraj/asl-dataset`
   (mucho más chico). Es barato y es lo único que permite un test set honesto.

Layout esperado: una carpeta por clase dentro de cada raíz. El script tolera un nivel extra de
anidamiento (los zips de Kaggle suelen traer `asl_alphabet_train/asl_alphabet_train/A/...`) y
normaliza los nombres de clase a minúsculas.

---

## 9. El pipeline: cuatro pasos

```bash
# 1. Landmarks desde las fotos originales (entorno de INFERENCIA — usa mediapipe)
./.venv-infer/bin/python prepare_dataset.py \
    --source alphabet=~/datasets/asl_alphabet_train \
    --source ayuraj=~/datasets/asl_dataset

# 2. Mirar qué ve la red ANTES de entrenar (cualquiera de los dos entornos)
./.venv-infer/bin/python check_render.py
./.venv-infer/bin/python check_render.py --augment

# 3. Entrenar (entorno de ENTRENAMIENTO)
./.venv-train/bin/python train_model.py --landmarks landmarks.npz \
    --epochs 40 --test-dataset ayuraj

# 4. Métricas honestas
./.venv-train/bin/python evaluate.py

# 5. Exportar para inferencia, con verificación de paridad
./.venv-train/bin/python export_onnx.py

# 6. Probar en vivo (entorno de INFERENCIA)
./.venv-infer/bin/python realtime_translator.py --activations
```

### Flags disponibles

```
prepare_dataset.py  --source NAME=PATH (repetible, obligatorio)  --out landmarks.npz
                    --classes a,b,...,z   --pad 0.25   --min-confidence 0.3
                    --workers <cpu/2>     --limit-per-class 0

train_model.py      --landmarks landmarks.npz   --out asl_cnn_model.keras
                    --input-size 96   --batch-size 128   --epochs 40
                    --val-fraction 0.2   --test-dataset ""   --seed 123
                    --mixed-precision

evaluate.py         --model asl_cnn_model.keras   --split split.npz   --landmarks ""

export_onnx.py      --model asl_cnn_model.keras   --out asl_cnn_model.onnx
                    --opset 15   --samples 200

check_render.py     --landmarks   --out docs/render-check.png   --size 96
                    --tile 96   --columns 13   --augment   --class a   --count 24

realtime_translator.py  --model   --classes   --camera 0
                        --activations   --debug-motion
```

### Consejo: probar el pipeline en chico primero

```bash
./.venv-infer/bin/python prepare_dataset.py --source alphabet=~/datasets/asl_alphabet_train \
    --limit-per-class 50 --out landmarks-small.npz
./.venv-train/bin/python train_model.py --landmarks landmarks-small.npz --epochs 3
```

Confirma que todo el circuito anda antes de gastar la corrida larga.

---

## 10. Verificación — en orden de importancia

1. **El contact sheet contra la ventana en vivo.** Correr `check_render.py`, abrir
   `docs/render-check.png`, y compararlo contra la ventana `AI View (Skeleton)` del traductor.
   **Tienen que verse iguales.** Es la prueba de que la brecha de dominio se cerró; si fallan, nada
   de lo demás importa. Qué mirar: manos centradas, todas del mismo tamaño, colores por dedo
   consistentes (pulgar crema, índice violeta, medio amarillo, anular verde, meñique azul,
   articulaciones de la palma rojas), ninguna mano cortada en el borde.

2. **Tasa de detección de MediaPipe.** `prepare_dataset.py` la reporta por clase y avisa si alguna
   baja del 60%. Esperable 85-95%. Una clase muy por debajo del resto hay que saberlo antes de
   entrenar, no después.

3. **Las curvas de entrenamiento.** `docs/training-accuracy.png`. Entrenamiento y validación tienen
   que ir juntas; una brecha que se abre significa que está memorizando.

4. **La matriz de confusión.** Las confusiones que queden deberían ser entre letras que
   genuinamente comparten forma de mano — **M, N, S y T** son todas un puño con el pulgar en
   distinto lugar. Eso es señal de que el modelo se comporta razonablemente. Cualquier otra cosa
   grande en la matriz merece investigación.

5. **Paridad ONNX.** `export_onnx.py` falla solo si la diferencia máxima supera 1e-4. En las pruebas
   sintéticas dio ~6.7e-06.

6. **La prueba en vivo.** Con la mano quieta la predicción **tiene que quedarse fija** — ese era el
   síntoma original. Después: deletrear una palabra completa, y probar J y Z.

7. **Calibrar J/Z.** Correr `realtime_translator.py --debug-motion`, hacer las señas, y mirar los
   valores medidos contra los umbrales en pantalla. Ajustar las constantes arriba de `motion.py`.
   Los valores actuales salieron de trayectorias sintéticas y es esperable que necesiten ajuste con
   una mano real y una cámara real.

---

## 11. Trampas conocidas (ya resueltas — no reintroducir)

### `BatchNormalization(momentum=0.99)` colapsa el modelo

El default de Keras asume corridas largas. Con pocos cientos de pasos las estadísticas móviles quedan
a medio converger, y con siete capas de normalización apiladas el error se compone hasta que **el
modelo predice una sola clase en inferencia**, aunque el accuracy de entrenamiento suba normal.

Peor: eso envenena a `EarlyStopping`, que ve validación clavada en el azar y elige la **época 1**
como la mejor.

Apareció al probar: validación clavada en 3.85% (= 1/26 exacto) con entrenamiento en 52%. Con
`momentum=0.9` pasó a 92%. **Ya está arreglado en `train_model.py`; no volver a 0.99.**

### `np.argsort` sobre claves compuestas

`natural_key()` devuelve tuplas. Pasarle una lista de tuplas a `np.argsort` la convierte en array 2D
y ordena por el eje equivocado, devolviendo índices 2D. Se usa `sorted()` de Python. Ya arreglado.

### La aumentación no debe recortar articulaciones sueltas

Un `np.clip` por punto dobla los dedos en vez de mover la mano. `augment_landmarks()` reescala y
desplaza el esqueleto **completo** para que entre en el lienzo. Verificado: 20.000 aumentaciones
quedan dentro de `[0,1]` con deformación relativa < 3.4%.

### OneDrive

El dataset (105k archivos) vivía dentro de OneDrive, que lo sincroniza y hace el I/O lentísimo — fue
el motivo original del `cache()` en disco. Guardar landmarks (~20 MB) en vez de imágenes elimina el
problema de raíz, pero igual conviene tener el proyecto fuera de cualquier carpeta sincronizada.

---

## 12. Qué esperar de los números

**No esperar 97.9%.** Ese número venía de fuga de datos. `evaluate.py` devuelve dos accuracies y la
brecha entre ellas es lo interesante:

- **validación** — bloque contiguo reservado de las mismas grabaciones. Mide cuánto aprendió estas
  manos en particular.
- **test** — un dataset entero que nunca vio, idealmente grabado por otra gente. **Este es el que
  predice el comportamiento en vivo**, y normalmente es bastante más bajo. Es el honesto.

Una brecha grande entre los dos no es un fracaso: es información. Significa que el modelo aprendió al
sujeto del dataset principal y no la seña en abstracto, y la respuesta es más diversidad de sujetos,
no más épocas.

---

## 13. Lo que falta

1. **Correr el pipeline completo con datos reales** (§9).
2. **Actualizar el README con las métricas reales.** Ahora mismo el README describe el diseño y
   explica cómo leer los resultados, pero **no afirma ningún porcentaje**, a propósito: no había
   modelo entrenado que los respaldara. Al terminar el paso 4, agregar los dos números.
3. **Reemplazar `asl_cnn_model.onnx`** (lo hace `export_onnx.py`) y confirmar que `class_names.txt`
   quedó con 26 clases (lo reescribe `train_model.py`).
4. **Calibrar los umbrales de J/Z** contra una mano real (§10, punto 7).
5. **Considerar borrar `asl_cnn_model.h5`** del repo: es el modelo viejo de 36 clases en formato
   legacy de Keras, ya sin nada que lo use. Está rastreado en git.
6. **Regenerar `arquitectura_3d.png` / `infografia_educativa.png`** si se usan para la presentación.
   `generar_arquitectura.py` ya fue actualizado a 26 clases y la arquitectura nueva; los `.png` no se
   regeneraron. Ambos scripts están en `.gitignore` (assets locales).

---

## 14. Decisiones ya tomadas — no re-litigar

El usuario las eligió explícitamente en la sesión anterior:

- **26 clases, solo letras A–Z.** Los dígitos se descartaron. Motivo doble: datos muy escasos
  (62-113 archivos por dígito contra 1887-5524 por letra; la clase `0` tenía solo 34 imágenes únicas
  detrás de 62 archivos) y ambigüedad genuina — como seña **estática**, `0≡O`, `2≡V`, `6≡W` y `9≡F`
  son la misma forma de mano.
- **Reprocesar desde las fotos originales con MediaPipe**, en lugar de reutilizar los esqueletos ya
  renderizados.
- **Sin captura propia por webcam.** Se descartó agregar un `collect_samples.py`. Consecuencia
  directa: J y Z tienen que resolverse por reglas, porque no hay secuencias grabadas para entrenar un
  modelo temporal. Si en algún momento se quiere la versión aprendida, hace falta grabar secuencias
  primero.
- **Alcance:** métricas serias, deletreo de palabras, J/Z dinámicas, y setup GPU en Linux. Los cuatro
  están implementados.

---

## 15. Referencia rápida de la API

```python
import preprocessing as pp

# MediaPipe → píxeles reales (21, 2). Convertir a píxeles ANTES de medir el bbox es lo
# que mantiene el aspecto honesto.
points_px = pp.landmarks_to_pixels(hand_landmarks, frame_w, frame_h)

# Centrar y escalar al 70% del lienzo. Entra y sale en el mismo espacio de unidades.
norm = pp.normalize_landmarks(points_px)          # (21, 2) en [0, 1]

# Dibujar. Devuelve RGB genuino, listo para el modelo.
image = pp.render_skeleton(norm, size=96)         # (96, 96, 3) uint8 RGB

# Aumentar: rotación ±15°, escala 0.9-1.1, traslación ±5%, jitter, espejado.
aug = pp.augment_landmarks(norm, rng)

# Atajo para inferencia en vivo: hace las tres cosas de arriba.
norm, image = pp.preprocess_hand(hand_landmarks, w, h)

# Agregar el eje de batch. El modelo tiene su propio Rescaling, los píxeles van 0-255.
batch = pp.model_input(image)
```

```python
from motion import MotionTracker

tracker = MotionTracker()
tracker.update(points_px)        # píxeles CRUDOS, no normalizados
letter = tracker.classify("i")   # 'j', 'z', o None
tracker.is_moving()              # para no confirmar letras a mitad de un gesto
tracker.debug_lines("i")         # lecturas en vivo para calibrar los umbrales
```

### Salidas del ONNX exportado

```
logits          (N, 26)         scores crudos; el softmax lo hace quien llame
conv1_relu      (N, H, W, 32)   primer bloque conv, alimenta el mosaico de filtros
dense_features  (N, 128)        capa oculta, alimenta la grilla de neuronas
```

Nombres estables, garantizados por `export_onnx.py`, que además empareja las salidas **por valor**
(no por orden) antes de renombrarlas, así un tap mal etiquetado no pasa silenciosamente.

---

## 16. Cómo se probó todo esto sin GPU ni datasets

Por si hace falta reproducir o extender las pruebas: se creó un venv aislado con `tensorflow-cpu`, se
generó un `landmarks.npz` sintético (26 clases, deformación fija por clase más ruido, dos datasets
para poder ejercitar `--test-dataset`), y se corrió la cadena completa
`prepare → train → evaluate → export → traductor`.

Resultados de esa corrida sintética: entrenamiento converge (validación 91% en 6 épocas), split sin
fuga, matriz de confusión generada, exportación ONNX con paridad de 6.7e-06, y el traductor
consumiendo el modelo nuevo de punta a punta. J y Z se probaron con trayectorias sintéticas; el
buffer de deletreo se probó en todos sus caminos (confirmación, anti-repetición, confianza baja,
movimiento, espacio, backspace, repeat, clear).

**Esas pruebas validan la mecánica, no la calidad del modelo.** La calidad solo se sabe con datos
reales.
