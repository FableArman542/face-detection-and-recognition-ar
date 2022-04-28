import cv2
import time
import numpy as np
from PIL import Image
from math import cos, sin
# import keyboard
from utils import euclidean_distance, get_left_and_right_eyes, get_angle, rotate_point, resize_image
from eigenface import Eigenfaces
from NNClassifier import Classifier


eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

def put_glass(glass, frame, x, y, width, height):
    hat_width = width + 1
    hat_height = int(.5 * height) + 1

    glass = cv2.resize(glass, (hat_width, hat_height))

    for i in range(hat_height):
        for j in range(hat_width):
            for k in range(3):
                if glass[i][j][k] < 235:
                    frame[y + i - int(-0.20 * height)][x + j][k] = glass[i][j][k]

def put_crown(crown, frame, x, y, width, height):
    crown_width = width+1
    crown_height = 50
    crown = cv2.resize(crown, (crown_width, crown_height))

    a = x
    b = y
    img_2_shape = crown.shape
    roi = frame[a:img_2_shape[0]+a, b:img_2_shape[1]+b]

    crown2gray = cv2.cvtColor(crown, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(crown2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Get background of crown
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Get foreground
    img2_fg = cv2.bitwise_and(crown, crown, mask=mask)

    # Put moon in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    frame[a:img_2_shape[0]+a, b:img_2_shape[1]+b] = dst

    # cv2.imshow('img1_bg', img1_bg)
    # cv2.imshow('img2_fg', img2_fg)


def normalize_face(box, gray, img, w, h):
    rotated = None
    (start_x, start_y, end_x, end_y) = box.astype('int')
    new_img_g = gray[start_y:end_y, start_x:end_x]
    new_img_c = img[start_y:end_y, start_x:end_x]

    # ey = face_cascade.detectMultiScale(gray, 1.09, 7)
    # frame=None
    # for (x, y, w, h) in ey:
    #     print(w, h)
    # width = int(euclidean_distance((start_x, 0), (end_x, 0)))
    # height = int(euclidean_distance((start_y, 0), (end_y, 0)))
    # put_glass(glasses, img, start_x, start_y, width, height)

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
        left_eye, left_eye_center, right_eye, right_eye_center = get_left_and_right_eyes(eyes_coordinates)

        left_eye_x_c = left_eye_center[0]; left_eye_y_c = left_eye_center[1]
        right_eye_x_c = right_eye_center[0]; right_eye_y_c = right_eye_center[1]

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
        scale = d / distance_between_eyes
        rotated = resize_image(rotated, scale_factor=scale)
        # print("distance", distance_between_eyes)
        new_left_eye = new_left_eye * scale
        new_right_eye = new_right_eye * scale
        face_start_x = int(new_left_eye[0]) - 16
        face_start_y = int(new_left_eye[1]) - 24
        face_end_x = (int(new_left_eye[0]) - 16) + 46
        face_end_y = (int(new_left_eye[1]) - 24) + 56

        rotated = rotated[face_start_y:face_end_y, face_start_x:face_end_x]

    return rotated
