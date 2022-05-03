import os
import cv2
import time
import numpy as np
from PIL import Image
from math import cos, sin
from utils import euclidean_distance, get_left_and_right_eyes, get_angle, rotate_point, resize_image
from eigenface import Eigenfaces
from NNClassifier import Classifier
from normalizer import normalize_face
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

prototxt = 'models/deploy.prototxt.txt'
caffemodel = 'models/res10_300x300_ssd_iter_140000.caffemodel'

model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

rotated = None
c = "5"
img = cv2.imread(os.path.join("resources/", c+".jpg"))
h, w, channels = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

columns = img.shape[1]
lines = img.shape[0]

blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
model.setInput(blob)
detections = model.forward()
perf_stats = model.getPerfProfile()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > .5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        rotated = normalize_face(box, gray, img, w, h)
        if rotated is not None and rotated.shape[0] == 56 and rotated.shape[1] == 46:
            cv2.imshow('rotated', rotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("resources/arman/face_new"+c+".jpg", rotated)