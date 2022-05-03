import cv2
from utils import euclidean_distance

def add_crown(crown, frame, x, y, width, height):
    crown_width = width + 1
    crown_height = int(0.4*height)
    crown = cv2.resize(crown, (crown_width, crown_height))

    a = y-crown_height
    b = x
    img_2_shape = crown.shape

    frame_w = frame.shape[1]
    frame_h = frame.shape[0]

    check_clipping(a, b, frame_w, frame_h)
    masking(a, b, crown, frame)

def add_face_tatoo(tattoo, frame, x, y, width, height):
    tattoo_width = int(width/4)
    tattoo_height = int(tattoo_width)
    print((tattoo_width, tattoo_height))
    tattoo = cv2.resize(tattoo, (tattoo_width, tattoo_height))

    center_x = int(x+euclidean_distance((x, 0), (x+width, 0))/2)
    center_y = int(y+euclidean_distance((y,0), (y+height, 0))/2)

    frame_w = frame.shape[1]
    frame_h = frame.shape[0]

    a = center_y
    b = center_x + 25

    check_clipping(a, b, frame_w, frame_h)
    masking(a, b, tattoo, frame)


def add_pig_nose(nose, frame, x, y, width, height):
    nose_width = int(width/4)
    nose_height = int(nose_width)
    nose = cv2.resize(nose, (nose_width, nose_height))

    center_x = int(x+euclidean_distance((x, 0), (x+width, 0))/2)
    center_y = int(y+euclidean_distance((y,0), (y+height, 0))/2)

    frame_w = frame.shape[1]
    frame_h = frame.shape[0]

    a = center_y-int(nose_height/2)+15
    b = center_x-int(nose_width/2)

    check_clipping(a, b, frame_w, frame_h)
    masking(a, b, nose, frame)

def check_clipping(x, y, w, h):
    if x < 0: x = 0
    elif x > w: x = w
    if y < 0: y = 0
    elif y > h: y = h

def masking(a, b, virtual_object, frame):
    try:
        img_2_shape = virtual_object.shape
        roi = frame[a:img_2_shape[0] + a, b:img_2_shape[1] + b]
        object2gray = cv2.cvtColor(virtual_object, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(object2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img2_fg = cv2.bitwise_and(virtual_object, virtual_object, mask=mask)
        dst = cv2.add(img1_bg, img2_fg)
        frame[a:img_2_shape[0] + a, b:img_2_shape[1] + b] = dst
    except:
        pass