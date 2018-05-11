import cv2


def process_image(img):
    hsv = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(src=hsv, lowerb=(29, 86, 6), upperb=(64, 255, 255))

    if cv2.countNonZero(mask) == 0:
        return False, False
    elif cv2.countNonZero(mask) > mask.shape[0] * mask.shape[1] * .60:
        return True, True

    return True, False


ss = cv2.imread("ss.png")

cv2.imshow("ss", ss)
cv2.waitKey(0)

print(process_image(ss))

ss1 = cv2.imread("ss1.png")

cv2.imshow("ss1", ss1)
cv2.waitKey(0)

print(process_image(ss1))

ss2 = cv2.imread("ss2.png")

cv2.imshow("ss2", ss2)
cv2.waitKey(0)

print(process_image(ss2))

