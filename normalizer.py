import cv2
from utils import euclidean_distance, get_left_and_right_eyes, get_angle, rotate_point, resize_image

eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

def normalize_face(box, gray, img, w, h):
    rotated = None
    (start_x, start_y, end_x, end_y) = box.astype('int')
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
        left_eye, left_eye_center, right_eye, right_eye_center = get_left_and_right_eyes(eyes_coordinates)

        left_eye_x_c = left_eye_center[0];
        left_eye_y_c = left_eye_center[1]
        right_eye_x_c = right_eye_center[0];
        right_eye_y_c = right_eye_center[1]

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