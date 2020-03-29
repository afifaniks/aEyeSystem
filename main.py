# In[1]
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils

# plt.style.use("ggplot")
# get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import regularizers
from keras_unet.metrics import iou, iou_thresholded, dice_coef
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from segmentor import FootpathSegmentor
from object_detector import ObjectDetector
from visualizer import Visualizer
from dist_calculator import DistanceCalculator

# load the best model
model = FootpathSegmentor(weight_path='weights/footpath_6k_blur.h5')
detector = ObjectDetector("weights/yolov3.weights", "cfg/yolov3.cfg")
dist_calc = DistanceCalculator("weights/distance_model.pkl")
visualizer = Visualizer()


# In[2]
def calculate_pixel_dist(left_borders, right_borders, inch_dist_per_pixel, mid_point):
    closest_left_border = np.max(left_borders)
    closest_right_border = np.min(right_borders)

    #     print(closest_left_border, closest_right_border)
    left_border_distance = (mid_point - closest_left_border) * inch_dist_per_pixel
    right_border_distance = (closest_right_border - mid_point) * inch_dist_per_pixel
    #     print(left_border_distance, right_border_distance)

    return [closest_left_border, closest_right_border, left_border_distance, right_border_distance]


def generate_output_frame(frame, video=False):
    line_border = 1
    mid_pt = 128

    if video:
        frame = imutils.rotate(frame, 270)  # Frame gets auto rotated, should be changed as per need

    frame = resize(frame, (256, 256, 3), mode='constant', preserve_range=False)

    segmentation_frame = frame.copy()
    detection_frame = frame.copy()
    frame_to_show = frame.copy()

    frame_mask = model.inference(array_to_img(segmentation_frame))

    # TODO: DONT DELETE
    # safe region
    # safe_reg = np.array([[[85, 250], [165, 250], [155, safe_region_y_axis], [95, safe_region_y_axis]]], dtype=np.int32)
    # frame_to_show = cv2.polylines(frame_to_show, [safe_reg], True, (255, 200, 40), 2)
    # TODO: DONT DELETE END

    frame_to_show = visualizer.draw_contour(frame_to_show, frame_mask)

    detection_boxes = detector.detect(detection_frame)

    frame_to_show = visualizer.draw_detection_boxes(frame_to_show, detection_boxes)

    return frame_to_show


# In[3]
image_data = cv2.imread('test/data (3).jpg')
# image_data = imutils.rotate(img_to_array(image_data), 270)
# print(type(img_to_array(image_data)))
plt.imshow(array_to_img(image_data))
plt.show()

binary_mask = model.inference(image_data)
plt.imshow(binary_mask)
plt.show()

dat = image_data
final_out = generate_output_frame(dat)
plt.imshow(final_out)
plt.show()

# In[ ]:


# # Video

# In[16]:


from PIL import Image

import numpy as np

# In[17]:


cap = cv2.VideoCapture('test/sample2.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)
if (cap.isOpened() == False):
    print("Error opening video  file")

# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3))
# font = cv2.FONT_HERSHEY_PLAIN

frame_count = 0
# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        # Display the resulting frame
        if frame_count == int(fps/5):  # Showing 3 images per frame
            frame_count = 0

            #             frame_to_show = generate_output_frame(frame, video=True)
            #             frame_to_show = object_detection(frame.copy(), net)
            frame_to_show = generate_output_frame(frame, video=True)

            cv2.imshow('aEye', frame_to_show)
        else:
            frame_count += 1

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()


