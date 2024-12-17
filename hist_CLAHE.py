import numpy as np
import cv2 as cv

img = cv.imread('L2409050001.009.original.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

cv.imwrite('L2409050001.009.CLAHE2-8.png',cl1)