import cv2
import numpy as np
classes = ["persona", "bicicleta", "auto", "moto", "avion", "autobus", "tren", "camion", "bote", "semáforo", "hidrante", "parquímetro", "banca", "pajaro", "gato", "perro", "caballo", "oveja", "vaca", "elefante", "oso", "cebra", "jirafa", "mochila", "paraguas", "bolso", "corbata", "maleta", "frisbee", "patineta", "esquí", "snowboard", "pelota deportiva", "cometa", "baseball", "baseball guante", "skateboard", "tabla de surf", "raqueta de tenis", "botella", "vino de cristal", "taza", "tenedor", "cuchillo", "cuchara", "tazón", "plátano", "manzana", "sándwich", "naranja", "brócoli", "zanahoria", "hot dog", "pizza", "dona", "pastel", "silla", "sofá", "planta en maceta", "cama", "mesa de comedor", "inodoro", "televisor", "laptop", "ratón", "control remoto", "teclado", "teléfono celular", "microondas", "horno", "tostadora", "fregadero", "nevera", "libro", "reloj", "vajilla", "taza", "tenedor", "cuchillo", "cuchara", "tazón", "computadora", "encimera", "barra de labios", "lápiz labial", "lapicero", "cartera", "bolso de hombre", "pulsera", "collar", "anillo", "camisa", "blusa", "suéter", "chaqueta", "abrigo", "vestido", "mono", "falda", "pantalones", "sombrero", "zapatos deportivos", "gafas", "sombrero", "bolso"]

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y + 30), font, 1, color, 2)

    cv2.imshow("Camara", frame)

    if cv2.waitKey(1) == 27:  # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
