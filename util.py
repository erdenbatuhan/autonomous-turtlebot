import math
import cv2
import numpy as np


c = lambda a, b: to_precision(math.sqrt(a ** 2 + b ** 2), 2)
to_precision_all = lambda arr, precision: [to_precision(el, precision) for el in arr]
normalized = lambda arr: [float(i) / sum(arr) for i in arr]


def get_index_of(arr, item):
    for i in range(len(arr)):
        if arr[i] == item:
            return i

    return -1


def to_precision(number, precision):
    if precision == 0:
        return int(number)

    number *= 10. ** precision
    number = int(number)
    number /= 10. ** precision

    return number


def get_angle_between(p1, p2):
    y = p2["y"] - p1["y"]
    x = p2["x"] - p1["x"]

    return math.atan2(y, x)


def get_distance_between(p1, p2):
    terminal = False

    a, b = p2["x"] - p1["x"], p2["y"] - p1["y"]
    c = math.sqrt(a ** 2 + b ** 2)

    if c < .5:
        terminal = True

    return [to_precision(a, 2), to_precision(b, 2)], to_precision(c, 2), terminal


def flatten(state, state_dim):
    state_flattened = np.zeros(state_dim)
    last_element = len(state_flattened) - 1

    state_flattened[0] = state[0]
    state_flattened[1:last_element] = state[1].reshape(1, -1)
    state_flattened[last_element] = state[2]

    return state_flattened


def process_image(img):
    hsv = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(src=hsv, lowerb=(29, 86, 6), upperb=(64, 255, 255))

    mask = cv2.erode(src=mask, kernel=None, iterations=2)
    mask = cv2.dilate(src=mask, kernel=None, iterations=2)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    mask_center, img_center = 0., img.shape[1] / 2

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10.:
            M = cv2.moments(c)
            mask_center = M["m10"] / M["m00"]

            # cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)

    return int(img_center - mask_center)

