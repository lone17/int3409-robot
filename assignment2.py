import sys

import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = random.choice(plt.colormaps())


import ai2thor.controller

import keyboard

import yolov3_darknet as yolo
from topview import TrajectoryDrawer

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
    should_update = False
    last_event = controller.last_event
    rot = last_event.metadata['agent']['rotation']
    rot_y = rot['y']
    if rot_y == 360 or rot_y == -360:
        rot_y = 0
    if keyboard.is_pressed('w'):
        event = controller.step(dict(action='MoveAhead'))
        should_update = event.metadata['lastActionSuccess']
    elif keyboard.is_pressed('s'):
        event = controller.step(dict(action='MoveBack'))
        should_update = event.metadata['lastActionSuccess']
    elif keyboard.is_pressed('a'):
        event = controller.step(dict(action='MoveLeft'))
        should_update = event.metadata['lastActionSuccess']
    elif keyboard.is_pressed('d'):
        event = controller.step(dict(action='MoveRight'))
        should_update = event.metadata['lastActionSuccess']
    elif keyboard.is_pressed('up'):
        event = controller.step(dict(action='LookUp'))
        should_update = event.metadata['lastActionSuccess']
    elif keyboard.is_pressed('down'):
        event = controller.step(dict(action='LookDown'))
        should_update = event.metadata['lastActionSuccess']
    elif keyboard.is_pressed('left'):
        rot['y'] = rot_y - 10
        event = controller.step(dict(action='Rotate', rotation=rot))
        should_update = event.metadata['lastActionSuccess']
    elif keyboard.is_pressed('right'):
        rot['y'] = rot_y + 10
        event = controller.step(dict(action='Rotate', rotation=rot))
        should_update = event.metadata['lastActionSuccess']
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
        should_update = event.metadata['lastActionSuccess']
    else:
        event = last_event


    if should_update:
        trajectories, trajectories_semantic = drawer.draw(event.metadata['agent']['position'])
        return {
            'object detection': yolo.process(event.frame),
            'trajectories': trajectories,
            'trajectories on semantic map' : trajectories_semantic,
            # event.instance_segmentation_frame
        }
    else:
        return None


SCREEN_SIZE = 450
GRID_SIZE = 0.2

controller = ai2thor.controller.Controller()
controller.start(player_screen_width=SCREEN_SIZE, player_screen_height=SCREEN_SIZE)

# Kitchens:       FloorPlan1 - FloorPlan30
# Living rooms:   FloorPlan201 - FloorPlan230
# Bedrooms:       FloorPlan301 - FloorPlan330
# Bathrooms:      FloorPLan401 - FloorPlan430
controller.reset('FloorPlan30')

# gridSize specifies the coarseness of the grid that the agent navigates on
event = controller.step(dict(action='Initialize',
                             renderClassImage=True,
                             renderObjectImage=True,
                             makeAgentVisible=False,
                             gridSize=GRID_SIZE))

rot = event.metadata['agent']['rotation']
rot['y'] = 0
controller.step(dict(action='Rotate', rotation=rot))

drawer = TrajectoryDrawer(controller)

while True:
    frames = get_frames(controller)
    if frames is None:
        continue
    for title, img in frames.items():
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(title, img)
    cv2.waitKey(1)
