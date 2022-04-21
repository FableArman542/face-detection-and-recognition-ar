import cv2
import time
import numpy as np
from PIL import Image
from math import cos, sin
from utils import angle_between, rotate_image, euclidean_distance

prototxt = 'models/deploy.prototxt.txt'
caffemodel = 'models/res10_300x300_ssd_iter_140000.caffemodel'

# with open(prototxt, 'r') as f:
#     class_names = f.read().split('\n')
# print(class_names)
# colors = np.random.uniform(0, 255, size=(len(class_names), 3))


model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture(0)

eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

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
            (start_x, start_y, end_x, end_y) = box.astype('int')

            text = "{:.2f}%".format(confidence*100)

            new_img_g = gray[start_y:end_y, start_x:end_x]
            new_img_c = img[start_y:end_y, start_x:end_x]
            eyes = eye_cascade.detectMultiScale(new_img_g, 1.3, 5)

            count_two_eyes = 0
            eyes_coordinates = []
            eye1 = None
            eye2 = None
            for (ex, ey, ew, eh) in eyes:
                if count_two_eyes == 0:
                    eye1 = (ex, ey, ew, eh)
                elif count_two_eyes == 1:
                    eye2 = (ex, ey, ew, eh)
                else:
                    break
                cv2.rectangle(new_img_c, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eyes_coordinates.append((ex, ey, ew, eh))
                count_two_eyes += 1


            # Draw a line between the eyes
            if count_two_eyes >= 2:

                # Calculate left and right eyes
                left_eye = None
                right_eye = None

                if eyes_coordinates[0][0] < eyes_coordinates[1][0]:
                    left_eye = eyes_coordinates[0]
                    right_eye = eyes_coordinates[1]
                else:
                    left_eye = eyes_coordinates[1]
                    right_eye = eyes_coordinates[0]

                left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
                left_eye_x = left_eye_center[0]
                left_eye_y = left_eye_center[1]

                right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
                right_eye_x = right_eye_center[0]
                right_eye_y = right_eye_center[1]

                cv2.circle(new_img_c, left_eye_center, 2, (255, 0, 0), 2)
                cv2.circle(new_img_c, right_eye_center, 2, (255, 0, 0), 2)
                cv2.line(new_img_c, right_eye_center, left_eye_center, (67, 67, 67), 2)

                if left_eye_y < right_eye_y:
                    aux = (right_eye_x, left_eye_y)
                    direction = -1
                else:
                    aux = (left_eye_x, right_eye_y)
                    direction = 1

                a = euclidean_distance(left_eye_center, aux)
                b = euclidean_distance(right_eye_center, left_eye_center)
                c = euclidean_distance(right_eye_center, aux)

                cos_a = (b * b + c * c - a * a) / (2 * b * c)

                angle = np.degrees(np.arccos(cos_a))

                y = start_y - 10 if start_y - 10 > 10 else start_y + 10

                cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
                cv2.putText(img, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 3)

                if direction == -1:
                    angle = 90 - angle

                # Cortar a imagem
                new_img = Image.fromarray(img)
                new_img = np.array(new_img.rotate(direction * angle))#[start_y:end_y, start_x:end_x]

    if new_img is not None:
        cv2.imshow('rotated_img', new_img)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
