import cv2
import imutils
import numpy as np
import urllib.request

# Load YOLO model
net = cv2.dnn.readNet("yolov3_custom_last.weights", "yolov3_custom.cfg")

# Load class labels
with open("classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len
                                         (classes), 3))

# Load Haar Cascade for car detection
car_cascade = cv2.CascadeClassifier("cars.xml")

# Load red and green light images
green_light = cv2.imread("inp1.jpg")
green_light = cv2.resize(green_light, (100, 100))

red_light = cv2.imread("red.jpg")
red_light = cv2.resize(red_light, (100, 100))

# IP Camera URL
url = 'http://192.168.232.217/cam-hi.jpg'

try:
    while True:
        # Capture frame from IP camera
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)

        if img is None:
            print("Error: Unable to retrieve image.")
            continue

        img = imutils.resize(img, width=600)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect cars using Haar cascade
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)

        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        ambulance_detected = False

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

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if label.lower() == "ambulance":
                    ambulance_detected = True

        # Draw rectangles for detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Show traffic light based on ambulance detection
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_offset, y_offset = 480, 20

        if ambulance_detected:
            img[y_offset:y_offset + 100, x_offset:x_offset + 100] = green_light
            cv2.putText(img, "Ambulance Detected - Give Green Signal", (30, 50), font, 0.7, (0, 255, 0), 2)
        else:
            img[y_offset:y_offset + 100, x_offset:x_offset + 100] = red_light
            cv2.putText(img, "Normal Traffic - Red Signal", (30, 50), font, 0.7, (0, 0, 255), 2)

        cv2.imshow("Traffic Signal Detection", img)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Terminated by user.")

finally:
    cv2.destroyAllWindows()
