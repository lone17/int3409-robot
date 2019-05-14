import sys

import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = random.choice(plt.colormaps())


import ai2thor.controller

import keyboard

import yolov3_darknet as yolo

def adjust_gamma(img, gamma=0.3):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(img, table)

def edge_detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(img, 100, 10)

def img_gradient(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(img,cv2.CV_64F)

def draw_contour(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = edge_detect(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    colour = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 255))
    cv2.drawContours(img, contours, -1, colour, 3)
    return img

def process(img):
    # img = adjust_gamma(img)
    # img = edge_detect(img)
    # img = img_gradient(img)
    # img = draw_contour(img)
    img = yolo.process(img)
    return img

def get_frames(controller):
    last_event = controller.last_event
    rot = last_event.metadata['agent']['rotation']
    rot_y = rot['y']
    if rot_y == 360 or rot_y == -360:
        rot_y = 0
    if keyboard.is_pressed('w'):
        event = controller.step(dict(action='MoveAhead'))
    elif keyboard.is_pressed('s'):
        event = controller.step(dict(action='MoveBack'))
    elif keyboard.is_pressed('a'):
        event = controller.step(dict(action='MoveLeft'))
    elif keyboard.is_pressed('d'):
        event = controller.step(dict(action='MoveRight'))
    elif keyboard.is_pressed('up'):
        event = controller.step(dict(action='LookUp'))
    elif keyboard.is_pressed('down'):
        event = controller.step(dict(action='LookDown'))
    elif keyboard.is_pressed('left'):
        rot['y'] = rot_y - 10
        event = controller.step(dict(action='Rotate', rotation=rot))
    elif keyboard.is_pressed('right'):
        rot['y'] = rot_y + 10
        event = controller.step(dict(action='Rotate', rotation=rot))
    elif keyboard.is_pressed('f'):
        objects = [o for o in last_event.metadata['objects'] if o['visible'] and o['openable']]
        if len(objects) == 0:
            event = last_event
        objects.sort(key=lambda o: o['distance'])
        nearest_obj = objects[0]
        if nearest_obj['isopen']:
            event = controller.step(dict(action='CloseObject', objectId=nearest_obj['objectId']))
        else:
            event = controller.step(dict(action='OpenObject', objectId=nearest_obj['objectId']))
    else:
        event = last_event

    # topdown_event = controller.step({'action': 'ToggleMapView'})
    # controller.step({'action': 'ToggleMapView'})
    return (
        event.frame,
        # event.class_segmentation_frame,
        # event.instance_segmentation_frame
    )

controller = ai2thor.controller.Controller()
controller.start(player_screen_width=800, player_screen_height=800)

# Kitchens:       FloorPlan1 - FloorPlan30
# Living rooms:   FloorPlan201 - FloorPlan230
# Bedrooms:       FloorPlan301 - FloorPlan330
# Bathrooms:      FloorPLan401 - FloorPlan430
controller.reset('FloorPlan28')

# gridSize specifies the coarseness of the grid that the agent navigates on
event = controller.step(dict(action='Initialize',
                             # renderClassImage=True,
                             # renderObjectImage=True,
                             gridSize=0.25))

ax = []
im = []

num_frames = 1

for i in range(num_frames):
    tmp_ax = plt.subplot(1, num_frames, i+1)
    tmp_ax.axis('off')
    ax.append(tmp_ax)

frames = get_frames(controller)
# while None in frames:
#     frames = get_frames(controller)

for i in range(num_frames):
    img = frames[i]
    img = process(img)
    im.append(ax[i].imshow(img))

while True:
    frames = get_frames(controller)
    # if None in frames:
    for i in range(num_frames):
        img = frames[i]
        img = process(img)
        im[i].set_data(img)
    plt.pause(0.00001)
