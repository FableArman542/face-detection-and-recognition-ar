import cv2
import time
import numpy as np
from PIL import Image
from math import cos, sin
import keyboard
from utils import euclidean_distance, get_left_and_right_eyes, get_angle, rotate_point, resize_image
from eigenface import Eigenfaces
from NNClassifier import Classifier
from normalizer import normalize_face
from augmented_reality import add_crown, add_pig_nose, add_face_tatoo

tattoo=cv2.imread('objects/fenix.png')
crown=cv2.imread('objects/crown.png')
naruto_band=cv2.imread('objects/naruto.png')
nose = cv2.imread('objects/pig_nose.png')

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

prototxt = 'models/deploy.prototxt.txt'
caffemodel = 'models/res10_300x300_ssd_iter_140000.caffemodel'

model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture(0)

eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

classifier = Classifier(m=14, algorithm='fischer')

rotated = None
do = False
c=0
while True:
    ret, img = cap.read()
    h, w, channels = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    columns = img.shape[1]
    lines = img.shape[0]

    blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), (104.0, 177.0, 123.0), False, False)
    model.setInput(blob)
    detections = model.forward()

    perf_stats = model.getPerfProfile()

    new_img = None

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > .5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            rotated = normalize_face(box, gray, img, w, h)
            if rotated is not None and rotated.shape[0] == 56 and rotated.shape[1] == 46:
                if do:
                    print("Writing...")
                    cv2.imwrite("resources/arman/face_na_faculdade" + str(c) + ".jpg", rotated)
                    c += 1
                    do = False
                new_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                new_rotated = np.reshape(new_rotated, (56 * 46))
                vector = classifier.get_vector(new_rotated)
                predicted = classifier.predict(vector)
                # cv2.imshow('rotated_img', rotated)
                ey = face_cascade.detectMultiScale(gray, 1.09, 7)
                for (x, y, w, h) in ey:
                    if predicted == "Prof":
                        add_crown(crown, img, x, y, w, h)
                    elif predicted == "Arman":
                        add_pig_nose(nose, img, x, y, w, h)
                        # add_crown(crown, img, x, y, w, h)
                        # add_face_tatoo(tattoo, img, x, y, w, h)
                    else:
                        add_face_tatoo(tattoo, img, x, y, w, h)

    if keyboard.is_pressed("b"):
        print("'B' Pressed")
        do=True

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
