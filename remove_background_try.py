"""_summary_

Returns:
    _type_: _description_
"""

import os
import cv2
import numpy as np
from utils.chromo_cv_utils import Metaphaser


# Function to remove background, cells and other contaminants
def process_image(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Thresholding to create a mask
    # Assuming that the contaminants and cells are darker than the background
    _, mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Invert mask if necessary (if cells are lighter than background)
    # mask = cv2.bitwise_not(mask)

    # Apply the mask to get the foreground (remove background)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Apply additional processing if needed, e.g., morphology to remove small contaminants
    # Here we can use morphological operations like opening to remove small objects
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return result


SRC_DIR = "D:\\download\\removebackground"

for fn in os.listdir(SRC_DIR):
    # img = process_image(os.path.join(SRC_DIR, fn))
    # new_fn = fn.split(".")[0] + "_rmbk.png"
    # cv2.imwrite(os.path.join(SRC_DIR, new_fn), img)

    img = cv2.imread(os.path.join(SRC_DIR, fn))
    meta = Metaphaser(img)
    dst = meta.metaphase()
    new_fn = fn.split(".")[0] + "_rmbk.png"
    cv2.imwrite(os.path.join(SRC_DIR, new_fn), dst)
