""" Convert all images in a directory to grayscale.
"""

import os
import cv2


SRC_DIR = "D:\\tmp\\tmp"
IMG_EXTs = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"]

for fn in os.listdir(SRC_DIR):
    # 不是图片文件则跳过
    if os.path.splitext(fn)[1].lower() not in IMG_EXTs:
        continue
    img = cv2.imread(os.path.join(SRC_DIR, fn))
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # save grayscale image
    new_fn = fn.split(".")[0] + "_gray.png"
    cv2.imwrite(os.path.join(SRC_DIR, new_fn), gray)
