import cv2
import numpy as np


# Stack 4 images to have a better understanding of the velocity
# img_stack = np.stack((img, img, img, img), axis=1)
# print(img_stack.shape)


def preprocess_image(img):
	# sConvert the image to grayscale
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Delete the very inner information from the image
	img = np.reshape(img, (img.shape[0], img.shape[1]))

	# Resize the image
	img = cv2.resize(img, (80, 80))

	return img

def get_greens(img):
	try:
		hsv = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(src=hsv, lowerb=(29, 86, 6), upperb=(64, 255, 255))

		_mask = cv2.resize(mask, (80, 80))
	except cv2.error as e:
		return np.zeros((80, 80)), False, False

	if cv2.countNonZero(mask) <= 50:
		return np.zeros((80, 80)), False, False
	elif cv2.countNonZero(mask) > mask.shape[0] * mask.shape[1] * .60:
		return _mask, True, True

	return _mask, True, False