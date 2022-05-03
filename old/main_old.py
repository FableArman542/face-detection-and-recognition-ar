import cv2
import time
import numpy as np
from PIL import Image
from math import cos, sin
# import keyboard
from utils import euclidean_distance, get_left_and_right_eyes, get_angle, rotate_point, resize_image
from eigenface import Eigenfaces
from NNClassifier import Classifier

prototxt = 'models/deploy.prototxt.txt'
caffemodel = 'models/res10_300x300_ssd_iter_140000.caffemodel'

model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture(0)

eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

classifier = Classifier(m=10, algorithm='fischer')

rotated = None
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
            (start_x, start_y, end_x, end_y) = box.astype('int')

            text = "{:.2f}%".format(confidence * 100)

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
                eyes_coordinates.append((ex, ey, ew, eh))
                count_two_eyes += 1

            if count_two_eyes >= 2:

                # Calculate left and right eyes
                left_eye, left_eye_center, right_eye, right_eye_center = get_left_and_right_eyes(eyes_coordinates)

                left_eye_x_c = left_eye_center[0]; left_eye_y_c = left_eye_center[1]
                right_eye_x_c = right_eye_center[0]; right_eye_y_c = right_eye_center[1]

                # Draw a line between the eyes
                # cv2.line(new_img_c, right_eye_center, left_eye_center, (67, 67, 67), 2)

                # Get angle between eyes
                angle, d = get_angle(left_eye_center, right_eye_center)

                face_center = (start_x + euclidean_distance((start_x, 0), (end_x, 0)) / 2,
                               start_y + euclidean_distance((start_y, 0), (end_y, 0)) / 2)

                scale_center = (w / 2, h / 2)

                # Rotation matrix definition
                rotation_matrix = cv2.getRotationMatrix2D(scale_center, -d * angle, 1.)

                # Rotated Image with the angle
                rotated = cv2.warpAffine(img, rotation_matrix, (w, h))

                face_center_rotated = rotate_point(face_center, rotation_matrix)

                if right_eye is not None and left_eye is not None:
                    new_left_eye = rotate_point((start_x + left_eye_x_c, start_y + left_eye_y_c), rotation_matrix)

                    new_right_eye = rotate_point((start_x + right_eye_x_c, start_y + right_eye_y_c), rotation_matrix)

                distance_between_eyes = euclidean_distance(new_left_eye, new_right_eye)
                d = 15
                scale = d/distance_between_eyes
                rotated = resize_image(rotated, scale_factor=scale)

                # print("distance", distance_between_eyes)
                new_left_eye = new_left_eye*scale
                new_right_eye = new_right_eye*scale

                face_start_x = int(new_left_eye[0]) - 16
                face_start_y = int(new_left_eye[1]) - 24
                face_end_x = (int(new_left_eye[0]) - 16) + 46
                face_end_y = (int(new_left_eye[1]) - 24) + 56

                # cv2.rectangle(rotated, (face_start_x, face_start_y), (face_end_x, face_end_y), (0, 0, 255), 1)
                rotated = rotated[face_start_y:face_end_y, face_start_x:face_end_x]
                if rotated.shape[0] == 56 and rotated.shape[1] == 46:
                    new_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                    new_rotated = np.reshape(new_rotated, (56 * 46))
                    vector = classifier.get_vector(new_rotated)
                    classifier.predict(vector)
                # Show rotated points
                # cv2.circle(rotated, face_center_rotated, 2, (0, 0, 255), 2)
                # cv2.circle(rotated, new_left_eye.astype(int), 2, (0, 255, 0), 2)
                # cv2.circle(rotated, new_right_eye.astype(int), 2, (0, 255, 0), 2)

    do=False
    # if keyboard.is_pressed("b"):
    #     print("PRESSED")
        # do=True
    if rotated is not None:
        cv2.imshow('rotated_img', rotated)
        # if do:
        #     if rotated.shape[0]==56 and rotated.shape[1]==46:
        #         print("HEREEE")
        #         cv2.imwrite("resources/eu/face"+str(c)+".jpg", rotated)
        #         c+=1
        #         do=False
    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
