import cv2
import numpy as np

# Arquitectura del modelo de reconocimiento 
prototxt = "Model/deploy.prototxt.txt"
# Pesos del modelo en la ruta almacenada
model = "Model/res10_300x300_ssd_iter_140000.caffemodel"
# Modelo de puntos faciales de OpenCV
landmark_model = "Model/lbfmodel.yaml.txt"

# Cargar el modelo de detección de rostros
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Cargar el modelo de puntos clave faciales
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(landmark_model)

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)  # '0' indica la cámara predeterminada

while True:
    # Leer un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape
    image_resized = cv2.resize(frame, (300, 300))

    # Crear blob para reprocesarlo
    blob = cv2.dnn.blobFromImage(image_resized, 1.0, (300, 300), (104, 117, 123))

    # Detección y predicción de imágenes
    net.setInput(blob)
    detections = net.forward()

    # Preparar lista de rectángulos de rostros detectados
    faces = []
    for detection in detections[0][0]:
        if detection[2] > 0.5:  # Confianza mínima de 50%
            box = detection[3:7] * [width, height, width, height]
            x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # Formato (x, y, w, h) para cada cara detectada
            faces.append((x_start, y_start, x_end - x_start, y_end - y_start))

    # Convertir la lista 'faces' a un array de NumPy de la forma esperada
    if len(faces) > 0:
        faces_array = np.array(faces, dtype=np.int32)

        # Detectar puntos faciales
        _, landmarks = facemark.fit(frame, faces_array)

        # Dibujar puntos y líneas en los puntos clave faciales
        for landmark in landmarks:
            for i in range(0, len(landmark[0])):
                x, y = int(landmark[0][i][0]), int(landmark[0][i][1])  # Convertir a enteros
                # Dibujar punto
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                # Conectar puntos con líneas
                if i in range(1, len(landmark[0])):
                    x_prev, y_prev = int(landmark[0][i - 1][0]), int(landmark[0][i - 1][1])  # Convertir a enteros
                    cv2.line(frame, (x_prev, y_prev), (x, y), (255, 0, 0), 1)

    # Mostrar el frame con las detecciones
    cv2.imshow("Real-time Face Detection with Landmarks", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
