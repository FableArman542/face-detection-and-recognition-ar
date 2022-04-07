import cv2
import time
import numpy as np
from math import cos, sin
from utils import angle_between, draw_angled_rec, rotate_image

prototxt = 'models/deploy.prototxt.txt'
caffemodel = 'models/res10_300x300_ssd_iter_140000.caffemodel'

# with open(prototxt, 'r') as f:
#     class_names = f.read().split('\n')
# print(class_names)
# colors = np.random.uniform(0, 255, size=(len(class_names), 3))


model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture(0)

eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

img = cv2.imread("./resources/pedroMjorge1Norm.png")
h, w, channels = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

columns = img.shape[1]
lines = img.shape[0]

blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), (104.0, 177.0, 123.0), False, False)
model.setInput(blob)
detections = model.forward()

perf_stats = model.getPerfProfile()

print('Inference time, ms: %.2f' % (perf_stats[0] / cv2.getTickFrequency() * 1000))
new_img = None
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > .5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype('int')

        text = "{:.2f}%".format(confidence * 100)

        new_img_g = gray[start_y:end_y, start_x:end_y]
        new_img_c = img[start_y:end_y, start_x:end_y]
        eyes = eye_cascade.detectMultiScale(new_img_g, 1.3, 5)

        count_two_eyes = 0
        eyes_coordinates = []
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(new_img_c, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            # cv2.circle(new_img_c, (ex, ey), 2, (255, 0, 0), 1)
            eyes_coordinates.append((ex, ey))
            count_two_eyes += 1

            if count_two_eyes >= 2: break

            # Draw a line between the eyes
        if count_two_eyes >= 2:
            y = start_y - 10 if start_y - 10 > 10 else start_y + 10

            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            cv2.putText(img, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 3)

            cv2.line(new_img_c, eyes_coordinates[0], eyes_coordinates[1], (255, 0, 0), 2)
            angle = angle_between(eyes_coordinates[0], eyes_coordinates[1], 'd')

            new_img = rotate_image(img, angle)

if new_img is not None:
    cv2.imshow('rotated_img', new_img)
cv2.imshow('img', img)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()

