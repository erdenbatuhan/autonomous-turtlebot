import cv2
import numpy as np

import skimage as skimage
from skimage import transform, color, exposure

img = cv2.imread("./depth_img_raw.png")

img = skimage.color.rgb2gray(img)
img = skimage.transform.resize(img, (80, 80))
img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))  # Is it really necessary??

# Reshape the image
img = np.reshape(img, (1, img.shape[0], img.shape[1]))
print(img.shape)

# Normalize
img = img / 255.0

# Stack 4 images to have a better understanding of the velocity
img_stack = np.stack((img, img, img, img), axis=1)
print(img_stack.shape)
