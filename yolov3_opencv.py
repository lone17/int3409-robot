import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

model_dir = "/mnt/e/Code/Robot/thor-iqa-cvpr-2018/darknet_object_detection/yolo_weights/"

LABELS = open(os.path.join(model_dir, 'thor.names'), 'r').read().strip().split('\n')
COLOURS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="int")

net = cv2.dnn.readNetFromDarknet(
    os.path.join(model_dir, "yolov3-thor.cfg"),
    # os.path.join('/home/lone/darknet/cfg', "yolov3-tiny.cfg"),
    os.path.join(model_dir, "yolov3-thor_final.weights"))

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def get_detections(image, outputs, thresh):
    image_h, image_w = image.shape[:2]

    bboxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence < thresh:
                continue

            center_x, center_y, box_w, box_h = detection[0:4] * np.array([image_w, image_h, image_w, image_h])
            x = center_x - (box_w / 2)
            y = center_y - (box_h / 2)
            w = int(box_w)
            h = int(box_h)
            x = int(np.clip(x, 0, image_w - 1))
            y = int(np.clip(y, 0, image_h - 1))

            bboxes.append([x, y, w, h])
            confidences.append(float(confidence))
            classIDs.append(int(classID))

    return bboxes, confidences, classIDs

def draw_bboxes(image, detections):
    for box, score, classID in detections:
        x, y, w, h = box
        colour = COLOURS[classID].tolist()
        label = LABELS[classID]
        cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
        text = "{}: {:.4f}".format(label, score)
        cv2.putText(image, text, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    return image

def process(image):
    origin_image = image.copy()
    h, w = origin_image.shape[:2]
    ratio = h / 240
    image = cv2.resize(image, (int(w / ratio), int(h / ratio)))

    blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    bboxes, confidences, classIDs = get_detections(image, outputs, thresh=0.7)
    bboxes = [[int(i * ratio) for i in box] for box in bboxes]
    idxs = cv2.dnn.NMSBoxes(bboxes, confidences, score_threshold=0.7, nms_threshold=0.5)
    detections = [[bboxes[i], confidences[i], classIDs[i]] for i in idxs.flatten()]
    image = draw_bboxes(origin_image, detections)

    return image

if __name__ == '__main__':
    image = cv2.imread('person.jpg')
    image = process(image)
    plt.imshow(image)
    plt.show()
