""" Convert all images in a directory to grayscale.
"""

import os
import cv2


SRC_DIR = "D:\\tmp\\tmp"

for fn in os.listdir(SRC_DIR):
    img = cv2.imread(os.path.join(SRC_DIR, fn))
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # save grayscale image
    new_fn = fn.split(".")[0] + "_gray.png"
    cv2.imwrite(os.path.join(SRC_DIR, new_fn), gray)
