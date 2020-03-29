import cv2
import numpy as np
from skimage.transform import resize


class ObjectDetector:
    def __init__(self, weight_path, cfg_path):
        # Load Yolo
        self.net = cv2.dnn.readNet(weight_path, cfg_path)

        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame):
        frame = resize(frame, (256, 256, 3), mode='constant', preserve_range=False)
        height, width, channels = frame.shape
        #     frame = imutils.rotate(frame, 270)
        # Detecting objects
        blob = cv2.dnn.blobFromImage(np.float32(frame), 1, (256, 256), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                #             print(detection)
                scores = detection[5:]
                #             print(scores)
                class_id = np.argmax(scores)
                #             print(class_id)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    if x >= 0 and y >= 0:
                        #                     print(x, y, w, h)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        #     print(indexes)
        detection_data = []

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                detection_data.append([x, y, w, h, label, confidence])

        return detection_data



