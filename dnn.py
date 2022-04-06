import cv2
import time
import numpy as np
from math import cos, sin
from utils import angle_between, draw_angled_rec
from RotatedRectangle import RRect

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

    print('Inference time, ms: %.2f' % (perf_stats[0] / cv2.getTickFrequency() * 1000))

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > .5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype('int')

            text = "{:.2f}%".format(confidence*100)

            new_img_g = gray[start_y:end_y, start_x:end_y]
            new_img_c = img[start_y:end_y, start_x:end_y]
            eyes = eye_cascade.detectMultiScale(new_img_g, 1.3, 5)

            y = start_y - 10 if start_y - 10 > 10 else start_y + 10

            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            cv2.putText(img, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 3)

            count_two_eyes = 0
            eyes_coordinates = []
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(new_img_c, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                # cv2.circle(new_img_c, (ex, ey), 2, (255, 0, 0), 1)
                eyes_coordinates.append((ex, ey))
                count_two_eyes += 1

                if count_two_eyes >= 2:
                    break
            # Draw a line between the eyes
            if count_two_eyes >= 2:
                cv2.line(new_img_c, eyes_coordinates[0], eyes_coordinates[1], (255, 0, 0), 2)
                angle = angle_between(eyes_coordinates[0], eyes_coordinates[1], 'rad')
                theta = angle
                rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

                width = abs(end_x - start_x)
                height = abs(end_y - start_y)
                a = (start_x, start_y)
                b = (end_x, start_y)
                c = (start_x, end_y)
                d = (end_x, end_y)
                center = (int(width / 2) + a[0], int(height / 2) + a[1])

                rot_a = (int(np.dot(rot, a)[0]), int(np.dot(rot, a)[1]))
                rot_b = (int(np.dot(rot, b)[0]), int(np.dot(rot, b)[1]))
                rot_c = (int(np.dot(rot, c)[0]), int(np.dot(rot, c)[1]))
                rot_d = (int(np.dot(rot, d)[0]), int(np.dot(rot, d)[1]))
                print(rot_a)
                # vetoracentro = (center[0]-a[0], center[1]-a[1])

                # cv2.line(img, center, a, (0, 0, 255), 2)
                # cv2.line(img, center, b, (0, 0, 255), 2)
                # cv2.line(img, center, c, (0, 0, 255), 2)
                # cv2.line(img, center, d, (0, 0, 255), 2)

                cv2.line(img, center, rot_a, (0, 255, 0), 2)
                cv2.line(img, center, rot_b, (0, 255, 0), 2)
                cv2.line(img, center, rot_c, (0, 255, 0), 2)
                cv2.line(img, center, rot_d, (0, 255, 0), 2)

                # draw_angled_rec(a, b, c, d,
                #                 width=abs(end_x-start_x),
                #                 height=abs(end_y-start_y),
                #                 img=new_img_c)
                # print("Angle", angle)
                # rr = RRect((start_x, start_y), (abs(end_x-start_x), abs(end_y-start_y)), angle)
                # rr.draw(new_img_c)



    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
