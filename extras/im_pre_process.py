import cv2
import math
import numpy as np


GREEN_LOWER = (29, 86, 6)
GREEN_UPPER = (64, 255, 255)


def get_distance_to_center(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    mask_center, img_center = 0., img.shape[1] / 2

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10.:
            M = cv2.moments(c)
            mask_center = M["m10"] / M["m00"]

            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            return int(math.fabs(mask_center - img_center))

    return -1


if __name__ == "__main__":
    img = cv2.imread("img.png")
    print(get_distance_to_center(img))

    img = cv2.imread("photo.jpg")
    print(get_distance_to_center(img))

    cv2.imshow("img", img)
    cv2.waitKey(0)