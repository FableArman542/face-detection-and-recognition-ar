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

def angle_between(v1, v2):
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    a = dot_product/(mag_v1*mag_v2)
    angle = np.arccos(a)

    angle = np.degrees(angle)

    return angle

def rotate_image(image, angle):
    height, width = image.shape[:2]
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, M=rot_mat, dsize=(width, height))
    return result
