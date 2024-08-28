import cv2

#Arquitectura del modelo en la ruta almacenada
prototxt = "Model/deploy.prototxt.txt"

#Pesos del model en la ruta almacenada
model = "Model/res10_300x300_ssd_iter_140000.caffemodel"

#Cargar el modelo
net = cv2.dnn.readNetFromCaffe(prototxt, model)

#Leer la imagen y procesarla
image = cv2.imread("dnn_face_detecter/Foto3.jpeg")
height, width, _ = image.shape
image_resized = cv2.resize(image, (300, 300))

#Crear blob para reprocesarlo
blob = cv2.dnn.blobFromImage(image_resized, 1.0, (300, 300), (104, 117, 123))
print("blob.shape: ", blob.shape)
blob_to_show = cv2.merge([blob[0][0], blob[0][1], blob[0][2]])

#Deteccion y prediccion de imagenes
net.setInput(blob)
detections = net.forward()
print("detections.shape:", detections.shape)

for detections in detections[0][0]:
    print(detections)
    if detections[2] > 0.5:
        box = detections[3:7] * [width, height, width, height]
        x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(image, "Conf: {:.2f}".format(detections[2] * 100), (x_start, y_start -5), 1, 1.2, (0, 255, 255), 2)

cv2.imshow("Image", image)
#cv2.imshow("blob_to_show", blob_to_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
