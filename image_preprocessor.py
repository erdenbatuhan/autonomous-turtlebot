import cv2
import numpy as np


# Stack 4 images to have a better understanding of the velocity
# img_stack = np.stack((img, img, img, img), axis=1)
# print(img_stack.shape)


def preprocess_image(img):
    # Delete the very inner information from the image
    img = np.reshape(img, (img.shape[0], img.shape[1]))

    # Resize the image
    img = cv2.resize(img, (80, 80))

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if np.isnan(img[i][j]):
                img[i][j] = -1

    # Reshape the image
    img = np.array([np.array([img])])

    return img
