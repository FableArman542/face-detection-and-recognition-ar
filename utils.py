import numpy as np
import cv2
import math

def distance_between(a, b):
    return (((b[0] - a[0]) ** 2) + ((b[1] - b[0]) ** 2)) ** 0.5

def angle_between(a, b, t):
    # h = distance_between(a, b)
    # c = abs(a[0]-a[0])
    # return np.arccos(c/h)
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    if t == 'degrees':
        return math.degrees(angle)

    return angle

def draw_angled_rec(a, b, c, d, width, height, img):
    center = (int(width / 2) + a[0], int(height / 2) + a[1])

    # center_x = (center[0] - b[0], center[1] - b[1])
    cv2.line(img, center, a, (0, 0, 255), 2)
    # print("center", center, a)