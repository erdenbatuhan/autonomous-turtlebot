import cv2
import numpy as np

import skimage as skimage
from skimage import transform, color


# Stack 4 images to have a better understanding of the velocity
# img_stack = np.stack((img, img, img, img), axis=1)
# print(img_stack.shape)


def preprocess_image(img):
    gray = skimage.color.rgb2gray(img)
    img_resized = skimage.transform.resize(gray, (80, 80))

    cv2.imshow("state", img_resized)
    cv2.waitKey(0)

    # Reshape the image
    img_reshaped = np.reshape(img, (1, img_resized.shape[0], img_resized.shape[1]))

    # Normalize
    img_normalized = img_reshaped / 255.0

    return img_normalized
