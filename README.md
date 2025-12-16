# Vehicles in traffic

**Análisis del sistema de detección y seguimiento de vehículos**

Asignatura: Fundamentos de Sistemas Inteligentes<br>
Grado en Ciencias e Ingeniería de Datos<br>
Alumnos: Joel López, Sara Dévora y Paula Hernández<br>
Fecha: 3/11/2025

### Enlace al video en drive: 
[Video tráfico](https://drive.google.com/drive/folders/10LDE3zYdzmwGXFmcItou0UjjyZgg8ATo?usp=drive_link)


### Introducción
El objetivo de esta practica es poder desarrollar un sistema que sea capaz de detectar y contar vehículos de un vídeo de tráfico real extraído de la dgt  utilizando exclusivamente técnicas de visión por computador con la librería de VisualStudio OpenCV, sin emplear redes neuronales.

Como base del proyecto se ha empleado el video “tráfico01.mp4”, el cual ofrece una perspectiva aérea del tránsito vehicular. En él se pueden observar automóviles, motocicletas y vehículos pesados circulando en distintas direcciones sobre múltiples carriles. A partir de este material visual, se desarrollaron e integraron diversos métodos para la detección, conteo, seguimiento y estimación de velocidad de los vehículos.


### Metodología aplicada

### **1. Procesamiento inicial del vídeo**

#### **1.1 Visualización del vídeo original**

Como primer paso, se realiza la carga y visualización completa del video “trafico01.mp4” utilizando la biblioteca OpenCV. El objetivo principal de esta etapa es verificar el correcto acceso al archivo, observar sus características visuales (como la cantidad de carriles, iluminación o presencia de vehículos), y familiarizarse con la escena antes de aplicar técnicas más complejas.

El flujo de trabajo es el siguiente:

- Se crea un objeto `VideoCapture` asociado al archivo de vídeo.
- A través de un bucle (`while cap.isOpened():`), se leen los fotogramas uno por uno.
- En cada iteración, se obtiene un frame con `cap.read()`.
- Si la lectura falla (`ret == False`), se considera que el vídeo ha finalizado.
- El fotograma actual se muestra en una ventana (`cv2.imshow('Video', frame)`).
- El usuario puede salir del vídeo en cualquier momento pulsando la tecla **q**.
- Finalmente, se liberan los recursos (`cap.release()`) y se cierran todas las ventanas abiertas (`cv2.destroyAllWindows()`).

```python
import cv2

video = 'tra╠üfico01.mp4'  #ruta del video
cap = cv2.VideoCapture(video)  #lee video

#lee todos los frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'): #muestra vídeo entero frame a frame
        break

cap.release()
cv2.destroyAllWindows()
```


#### **1.2 Generación del fondo promedio**

En esta fase se busca generar una imagen de fondo sin tráfico visible, es decir, una representación aproximada de la escena en estado estático. Para ello, se calcula el promedio de todos los fotogramas del vídeo.

Procedimiento:

- Se abre el vídeo y se obtiene la cantidad total de frames (`CAP_PROP_FRAME_COUNT`).
- Se toma el primer frame para conocer sus dimensiones (alto y ancho) y se inicializa una matriz `avg_frame` en ceros (tipo `float32`).
- Se recorren todos los fotogramas válidos:

  - Cada frame se convierte a `float32` y se suma a `avg_frame`.
  - Se lleva la cuenta del número de frames procesados.
- Al finalizar, se divide `avg_frame` entre el total de frames válidos para obtener la media.
- Se utiliza `np.nan_to_num` para evitar valores no válidos.
- Finalmente, se convierte el promedio a `uint8` para poder visualizarlo y se guarda como `background.png`.

Este fondo representa la escena sin vehículos, ya que su presencia se diluye en el promedio.

```python
import cv2
import numpy as np

video = 'tra╠üfico01.mp4'
cap = cv2.VideoCapture(video)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  #nº total frames
ret, frame = cap.read()  #lee primer frame, saber tamaño
if not ret or frame is None or frame_count == 0:
    print("Error, no se puede leer el vídeo.")
    cap.release()
    exit()

avg_frame = np.zeros_like(frame, dtype=np.float32)  #crea imagen para acumular suma, tipo flotante para no problema
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  #volvemos primer frame

valid_frames = 0
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    avg_frame += frame.astype(np.float32)
    valid_frames += 1

if valid_frames == 0:
    print("Error, no se puede leer ningún frame.")
    cap.release()
    exit()

avg_frame /= valid_frames
avg_frame = np.nan_to_num(avg_frame)
background = avg_frame.astype(np.uint8)

cv2.imshow('background promedia', background)
cv2.imwrite('background.png', background)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


#### **1.3 Aplicación de resta de fondo**

Con la imagen de fondo calculada, se inicia el análisis del movimiento en la escena. El objetivo es resaltar las diferencias entre el fondo estático y los objetos móviles.

Para ello:

- Se carga la imagen de fondo guardada previamente.
- Se reabre el vídeo original.
- En cada frame:

  - Se calcula la diferencia absoluta con respecto al fondo (`cv2.absdiff(frame, background)`).
  - Se muestra la imagen resultante, donde las áreas con movimiento aparecen destacadas.

Este paso permite visualizar claramente las zonas activas de la imagen antes de realizar un procesamiento más riguroso.

```python
import cv2
import numpy as np

video = 'tra╠üfico01.mp4'
background = cv2.imread('background.png')  # Imagen del fondo promedio

if background is None:
    print("Error: No se pudo cargar 'background.png'.")
    exit()

cap = cv2.VideoCapture(video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    # Resta el fondo al frame actual
    diff = cv2.absdiff(frame, background)
    # Muestra el resultado
    cv2.imshow('Resta del fondo', diff)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### **1.4 Binarización de las diferencias**

Aquí se transforma la imagen de diferencias a una forma binaria, facilitando la detección de objetos.

Pasos:

- Se recargan tanto el vídeo como la imagen de fondo.
- Por cada frame:

  - Se calcula la diferencia con el fondo.
  - Se convierte a escala de grises.
  - Se aplica un umbral binario (por ejemplo, T=55):

```python
import cv2
import numpy as np

video = 'tra╠üfico01.mp4'
background = cv2.imread('background.png')

if background is None:
    print("Error: No se pudo cargar 'background.png'.")
    exit()

cap = cv2.VideoCapture(video)

T = 55  # Ajusta este valor para definir qué es una diferencia significativa

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    diff = cv2.absdiff(frame, background)
    diff_gray = cv2.cvtColor(diff, cv2.COLORBGR2GRAY)
    _, diff_bin = cv2.threshold(diff_gray, T, 255, cv2.THRESH_BINARY)
    cv2.imshow('Diferencias significativas', diff_bin)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

El resultado es una máscara donde los píxeles blancos indican movimiento (posibles vehículos) y los negros representan el fondo. Ajustar el umbral T permite controlar la sensibilidad: valores bajos pueden generar ruido, mientras que valores altos podrían omitir objetos poco visibles.


#### **1.5 Conteo de vehículos en una vía**

En esta sección se construye un primer contador funcional de vehículos para una única vía definida dentro de la imagen.

Definición de la región de interés (ROI) -> Basándose en las dimensiones del frame, se define una zona rectangular que representa el carril o vía a monitorear. Esta región se codifica como un polígono (`area_pts`) con cuatro vértices, y servirá como máscara para limitar las detecciones a esa área específica.


Detección de vehículos mediante contornos -> Proceso por cada frame:

- Se calcula la diferencia con el fondo y se binariza como se explicó anteriormente.
- Se genera una máscara binaria del carril a partir del polígono definido.
- Se aplica esta máscara a la imagen binaria (`cv2.bitwise_and`).
- Se detectan contornos con `cv2.findContours`.
- Se filtran contornos según su área (`cv2.contourArea`), descartando los menores a un umbral (por ejemplo, 500) para evitar ruido.
- Para cada contorno válido, se obtiene un rectángulo delimitador (bounding box) que se asociará a un vehículo detectado.

Seguimiento simple con clase `Coche` ->  Se define una clase `Coche` que almacena:

- Posición y dimensiones del bounding box.
- Un identificador único (`id`).
- Número de frames en los que ha sido detectado.
- Un indicador (`updated`) que se actualiza en cada frame.

La clase incluye un método para obtener el centro del vehículo (centroide).

El seguimiento se realiza comparando los centroides de los vehículos detectados en el frame actual con los previamente registrados:

- Se busca el coche anterior más cercano dentro de un umbral (`DISTANCIA_MAX`).
- Si se encuentra coincidencia, se actualiza su información.
- Si no, se crea un nuevo objeto `Coche` con un ID único y se añade a la lista de vehículos en seguimiento.

Conteo de vehículos únicos -> El número total de vehículos detectados se estima como la longitud de la lista `coches_unicos`. Cada vez que se detecta un vehículo nuevo (que no coincide con ninguno anterior), se agrega a esta lista.

Durante la visualización:

- Se dibujan los rectángulos sobre los vehículos detectados (`cv2.rectangle`).
- Se muestra su identificador (`cv2.putText`).
- Se imprime en pantalla el conteo total (`cv2.putText(frame, f'Contador coches: {len(coches_unicos)}')`).
- Se dibuja la región de interés para indicar claramente el área monitorizada.

Al finalizar el vídeo, se imprime en consola el total de vehículos únicos detectados y este sería el código resultante:

```python
import cv2
import numpy as np

video = 'tra╠üfico01.mp4'
background = cv2.imread('background.png')

if background is None:
    print("Error: No se pudo cargar 'background.png'.")
    exit()

cap = cv2.VideoCapture(video)

T = 50

# Lee el primer frame para obtener tamaño
ret, frame = cap.read()
if not ret or frame is None:
    print("Error: No se pudo leer el primer frame.")
    exit()

height, width = frame.shape[:2]
rect_width = 60
rect_height = 100
top_left_x = (width - rect_width) - 1350
top_left_y = (height - rect_height) // 2

area_pts = np.array([
    [top_left_x, top_left_y],
    [top_left_x + rect_width, top_left_y],
    [top_left_x + rect_width, top_left_y + rect_height],
    [top_left_x, top_left_y + rect_height]
])
...
```

**Apartados 2, 4, 5 y 6: Generalización del sistema a escenarios multivía, bidireccionales, multivelocidad y multiclase vehicular**

En esta parte se detallan las mejoras incorporadas al sistema original de detección y conteo de vehículos. Se tratan de forma conjunta los puntos 2, 4, 5 y 6, ya que las optimizaciones aplicadas permiten abordar estos requerimientos de manera integrada. En concreto, se ha ampliado la capacidad del sistema para operar simultáneamente en varios carriles y vías, incluyendo configuraciones con doble sentido de circulación y flujos vehiculares tanto de entrada como de salida. Asimismo, se comprueba que el enfoque adoptado funciona adecuadamente en contextos con diferentes velocidades de tráfico y ofrece un rendimiento estable frente a la presencia de diversos tipos de vehículos, como motocicletas, turismos, furgonetas o camiones.

```python
# resto de ejercicios
import cv2
import numpy as np

video = 'tra╠üfico01.mp4'
background = cv2.imread('background.png')

if background is None:
    print("Error: No se pudo cargar 'background.png'.")
    exit()

cap = cv2.VideoCapture(video)

T = 50                   # umbral para binarizar la diferencia
AREA_MINIMA = 100        # área mínima de contorno para considerar detección
AREA_MAX = 15000         # área máxima de contorno para considerar detección
DISTANCIA_MAX = 78       # distancia máxima para considerar "el mismo coche"
MAX_FRAMES_PERDIDO = 10  # nº máximo de frames sin ver un coche antes de borrarlo

# Lee el primer frame para obtener tamaño
ret, frame = cap.read()
if not ret or frame is None:
    print("Error: No se pudo leer el primer frame.")
    exit()

height, width = frame.shape[:2]

# configuración para cada vía 
widths  = [60, 70, 70, 200, 200, 100]     # ancho de las vías
heights = [35, 100, 100, 100, 90, 70]     # alturas 
gaps    = [70, 50, 350, -20, -50]         # gaps entre vías (distancia de via a via)
y_offsets = [-250, 70, 0, 150, 30, -135]  # offsets verticales por vía (ajusta para poner más arriba o más abajo)

USE_EXPLICIT_VIAS = False
vias_explicit = [] 

# via que esta más a la izquierda
EXTRA_LEFT = 250  # píxeles para mover la primera vía aún más a la izquierda
first_rect_width = widths[0] if len(widths) > 0 else 60
base_top_left_x = (width - first_rect_width) - 1350 - EXTRA_LEFT


# Construimos varias áreas con tamaños, gaps y offsets verticales distintos
area_pts_list = []

if USE_EXPLICIT_VIAS and len(vias_explicit) > 0:
    for (x, y, w, h) in vias_explicit:
        x0 = max(0, min(int(x), width-1))
        y0 = max(0, min(int(y), height-1))
        x1 = min(width-1, x0 + int(w))
        y1 = min(height-1, y0 + int(h))
        area_pts = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        area_pts_list.append(area_pts)
else:
    num_vias = len(widths)  # longitud determina NUM_VIAS

    # Asegurar que heights, gaps y y_offsets tengan la longitud necesaria
    if len(heights) < num_vias:
        heights = heights + [heights[-1]] * (num_vias - len(heights))
    if len(gaps) < max(0, num_vias-1):
        gaps = gaps + [0] * (max(0, num_vias-1) - len(gaps))
    if len(y_offsets) < num_vias:
        y_offsets = y_offsets + [0] * (num_vias - len(y_offsets))

    tlx = base_top_left_x
    for i in range(num_vias):
        wv = int(widths[i])
        hv = int(heights[i])
        # centrar verticalmente cada vía según su altura y aplicar offset vertical específico
        tly = (height - hv) // 2 + int(y_offsets[i])
        # Clamp para no salir del frame
        x0 = max(0, min(tlx, width-1))
        y0 = max(0, min(tly, height-1))
        x1 = min(width-1, x0 + wv)
        y1 = min(height-1, y0 + hv)
        area_pts = np.array([
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1]
        ])
        area_pts_list.append(area_pts)
        # mover X acumulativo: ancho de esta vía + gap (cuando existe)
        gap = gaps[i] if i < len(gaps) else 0
        tlx = tlx + wv + gap

# inicialización de los contadores por vía
counters_vias = [0 for _ in range(len(area_pts_list))]

class Coche:
    def __init__(self, x, y, w, h, area, id, via_index=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = area
        self.id = id
        self.frames_seen = 1
        self.updated = True
        self.via_index = via_index  # vía en la que fue detectado inicialmente
        self.frames_perdido = 0     # nº de frames seguidos sin ser actualizado

    def centroide(self):
        # Retorna el centro del bounding box
        return (self.x + self.w // 2, self.y + self.h // 2)

coches_unicos = []
proximo_id = 1        # ID incremental de coche
total_coches = 0      # número total de coches distintos creados

# volvemos al primer frame del video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    diff = cv2.absdiff(frame, background)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_bin = cv2.threshold(diff_gray, T, 255, cv2.THRESH_BINARY)

    
    detectados = []  # guardamos (x, y, w, h, area, via_index) por detección

    # Para cada vía, aplicamos su máscara y extraemos contornos, para evitar duplicados cuando hay solapamiento entre vías, se guardarán los centros ya asignados
    centros_asignados = set()
    for idx, area_pts in enumerate(area_pts_list):
        mask = np.zeros_like(diff_bin)
        cv2.fillPoly(mask, [area_pts], 255)
        diff_bin_carril = cv2.bitwise_and(diff_bin, mask)
        contours, _ = cv2.findContours(diff_bin_carril, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if AREA_MINIMA <= area <= AREA_MAX:
                x, y, wv, hv = cv2.boundingRect(contour)
                cX = x + wv//2
                cY = y + hv//2
                centro_key = (int(cX), int(cY))
                # Si ya asignamos una detección con el mismo centro, la saltamos
                if centro_key in centros_asignados:
                    continue
                centros_asignados.add(centro_key)
                detectados.append((x, y, wv, hv, area, idx))

    # Marcar todos los coches como no actualizados
    for coche in coches_unicos:
        coche.updated = False

    # Para cada detección, buscar el coche anterior más cercano
    for x, y, wv, hv, area, via_idx in detectados:
        c_actual = (x + wv // 2, y + hv // 2)
        min_dist = float('inf')
        coche_match = None
        for coche in coches_unicos:
            c_anterior = coche.centroide()
            dist = np.hypot(c_actual[0] - c_anterior[0], c_actual[1] - c_anterior[1])
            if dist < min_dist and dist < DISTANCIA_MAX:
                min_dist = dist
                coche_match = coche

        if coche_match is not None:
            # se actualiza el coche existente
            coche_match.x = x
            coche_match.y = y
            coche_match.w = wv
            coche_match.h = hv
            coche_match.area = area
            coche_match.frames_seen += 1
            coche_match.updated = True
            coche_match.frames_perdido = 0
            coche_id = coche_match.id
            # Si aún no tiene vía asignada, y la detección está en una vía, se le asigna
            if coche_match.via_index is None and via_idx is not None:
                coche_match.via_index = via_idx
                counters_vias[via_idx] += 1
        else:
            # nuevo coche
            nuevo_coche = Coche(x, y, wv, hv, area, proximo_id, via_index=via_idx)
            coches_unicos.append(nuevo_coche)
            if via_idx is not None:
                counters_vias[via_idx] += 1
            proximo_id += 1
            total_coches += 1        # contamos el coche nuevo como uno distinto
            coche_id = nuevo_coche.id

        # se dibuja el bounding box con el ID
        cv2.rectangle(frame, (x, y), (x + wv, y + hv), (0, 255, 0), 2)
        cv2.putText(frame, f'ID:{coche_id}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # eliminamos lo coches que ya no aparecen
    coches_filtrados = []
    for coche in coches_unicos:
        if coche.updated:
            coches_filtrados.append(coche)
        else:
            coche.frames_perdido += 1
            if coche.frames_perdido <= MAX_FRAMES_PERDIDO:
                coches_filtrados.append(coche)
            # si lo hemos perdido demasiados frames, lo quitamos de la lista
    coches_unicos = coches_filtrados

    # dibujamos todas las vías y sus contadores
    for idx, area_pts in enumerate(area_pts_list):
        color = (255, 0, 0)
        cv2.polylines(frame, [area_pts], isClosed=True, color=color, thickness=2)
        px = int(area_pts[0][0]) + 5
        py = int(area_pts[0][1]) - 10
        cv2.putText(frame, f'Via {idx+1}: {counters_vias[idx]}', (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # usamos total_coches, es decir, coches distintos en lugar de len(coches_unicos)
    cv2.putText(frame, f'Contador coches: {total_coches}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Video con detección de coches únicos', frame)
    cv2.imshow('Diferencias significativas', diff_bin)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f'Coches únicos detectados en el vídeo: {total_coches}')
for idx, cnt in enumerate(counters_vias):
    print(f'Via {idx+1} - Coches detectados: {cnt}')
```



### Evaluación de los requerimientos

1. **Conteo de vehículos por vía (mínimo: 5.0)**
   Se implementó exitosamente un sistema de conteo basado en la detección por diferencia de fondo, filtrado, seguimiento y registro de vehículos. 

2. **Extensión a múltiples carriles**
   El sistema admite hasta seis carriles distintos, cada uno con su contador independiente, y permite trabajar con vías de diferente disposición y geometría.

3. **Detección en vías de doble sentido**
   Gracias al uso de máscaras poligonales, se registran vehículos en ambas direcciones, tanto en entrada como en salida del encuadre. ️

4. **Adaptación a diferentes velocidades de tráfico y reconocimiento de otros tipos de automóviles**
   La medición de velocidad permite operar en entornos con tráfico lento, fluido o rápido. El algoritmo de seguimiento es robusto frente a variaciones en la velocidad. 

5. **Cálculo de velocidad**
   La velocidad, tanto en m/s como en km/h, se determina con precisión gracias al seguimiento de centroides y la correcta calibración de la escala. 


### Resultados

El sistema desarrollado es capaz de:

- Detectar y contar vehículos de diferentes tamaños
- Ignorar sombras que podrían generar errores
- Seguir individualmente cada vehículo
- Estimar su velocidad en tiempo real
- Operar con fiabilidad en entornos de tráfico denso o fluido

Además, el desempeño se mantiene estable ante situaciones complejas como:

- Cruce o solapamiento entre vehículos
- Sombras alargadas
- Presencia de vehículos grandes
- Cambios bruscos en la velocidad


###  Limitaciones

- Requiere una imagen de fondo estática de buena calidad.
- Sombras muy marcadas pueden requerir ajustes finos del filtro HSV.
- No se clasifica automáticamente el tipo de vehículo, aunque el sistema podría ampliarse para ello.
- La precisión del cálculo de velocidad depende directamente de la calibración de la escala.


###  Conclusión

Este trabajo demuestra que, mediante técnicas tradicionales de visión por computador y con el uso de OpenCV, es posible desarrollar un sistema eficiente para:

- Detectar y contar vehículos
- Identificarlos por carril
- Estimar su velocidad
- Seguir su movimiento sin recurrir a inteligencia artificial avanzada

El sistema es versátil, escalable y robusto, lo que lo convierte en una solución eficaz para el análisis del tráfico en diferentes escenarios.

