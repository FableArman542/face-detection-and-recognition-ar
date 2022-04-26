import numpy as np
import cv2


def distance_between(a, b):
    return (((b[0] - a[0]) ** 2) + ((b[1] - b[0]) ** 2)) ** 0.5


def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    return np.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def get_angle(left_eye_center, right_eye_center):
    left_eye_x_c = left_eye_center[0]
    left_eye_y_c = left_eye_center[1]

    right_eye_x_c = right_eye_center[0]
    right_eye_y_c = right_eye_center[1]

    if left_eye_y_c < right_eye_y_c:
        aux = (right_eye_x_c, left_eye_y_c)
        d = -1
    else:
        aux = (left_eye_x_c, right_eye_y_c)
        d = 1

    a = euclidean_distance(left_eye_center, aux)
    b = euclidean_distance(right_eye_center, left_eye_center)
    c = euclidean_distance(right_eye_center, aux)

    cos_a = (b * b + c * c - a * a) / (2 * b * c)
    angle = np.degrees(np.arccos(cos_a))

    if d == -1:
        angle = 90 - angle

    return angle, d


def get_left_and_right_eyes(eyes_coordinates):
    if eyes_coordinates[0][0] < eyes_coordinates[1][0]:
        left_eye = eyes_coordinates[0]
        right_eye = eyes_coordinates[1]
    else:
        left_eye = eyes_coordinates[1]
        right_eye = eyes_coordinates[0]

    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))

    return left_eye, left_eye_center, right_eye, right_eye_center


def rotate_point(point, rotation_matrix):
    return np.matmul(rotation_matrix, np.array([point[0], point[1], 1])).astype(int)

def resize_image(img, scale_factor):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)

    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
