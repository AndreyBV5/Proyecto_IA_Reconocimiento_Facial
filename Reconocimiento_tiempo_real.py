import cv2

# Arquitectura del modelo en la ruta almacenada
prototxt = "Model/deploy.prototxt.txt"

# Pesos del modelo en la ruta almacenada
model = "Model/res10_300x300_ssd_iter_140000.caffemodel"

# Cargar el modelo
net = cv2.dnn.readNetFromCaffe(prototxt, model)

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

    for detection in detections[0][0]:
        if detection[2] > 0.5:
            box = detection[3:7] * [width, height, width, height]
            x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(frame, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (0, 255, 255), 2)

    # Mostrar el frame con las detecciones
    cv2.imshow("Real-time Face Detection", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
