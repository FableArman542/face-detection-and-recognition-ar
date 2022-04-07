import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    new_img = None
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        new_img_g = gray[y:y+h, x:x+w]
        new_img_c = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(new_img_g, 1.3, 5)

        start_x = x
        start_y = y
        end_x = x+w
        end_y = y+h

        count_two_eyes = 0
        eyes_coordinates = []
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(new_img_c, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
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
    k = cv2.waitKey(30) & 0xff
    if k == 27: break