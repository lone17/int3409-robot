import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

import darknet as dn
dn.set_gpu(0)

model_dir = b"/mnt/e/Code/Robot/thor-iqa-cvpr-2018/darknet_object_detection/yolo_weights/"

net = dn.load_net(
    os.path.join(model_dir, b"yolov3-thor.cfg"),
    # os.path.join(b'/home/lone/darknet/cfg', b"yolov3-tiny.cfg"),
    os.path.join(model_dir, b"yolov3-thor_final.weights"),
    0)

meta = dn.load_meta(os.path.join(model_dir, b"thor.data"))

LABELS = open(os.path.join(model_dir.decode('utf8'), 'thor.names'), 'r').read().strip().split('\n')
COLOURS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="int")

def draw_bboxes(image, detections):
    image_h, image_w = image.shape[:2]

    for label, score, box in detections:
        label = label.decode('utf8')
        center_x, center_y, box_w, box_h = box
        x1 = int(center_x - (box_w / 2))
        y1 = int(center_y - (box_h / 2))
        x2 = int(x1 + box_w)
        y2 = int(y1 + box_h)
        x1, x2 = np.clip([x1, x2], 0, image_w)
        y1, y2 = np.clip([y1, y2], 0, image_h)

        colour = COLOURS[LABELS.index(label)].tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        text = "{}: {:.4f}".format(label, score)
        cv2.putText(image, text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    return image

def process(image):
    origin_image = image.copy()
    h, w = origin_image.shape[:2]
    ratio = h / 288
    image = cv2.resize(image, (int(w / ratio), int(h / ratio)))

    detections = dn.detect(net, meta, image, thresh=0.7)
    for i, (label, score, box) in enumerate(detections):
        box = [int(i * ratio) for i in box]
        detections[i] = (label, score, box)
    image = draw_bboxes(origin_image, detections)

    return image

if __name__ == '__main__':
    image = cv2.imread('person.jpg')
    image = process(image)
    plt.imshow(image)
    plt.show()
