import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("L2409050001.009.original.png", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# 创建一个空的输出数组
normalized_img = np.zeros(img.shape, dtype=np.uint8)

# 归一化操作，范围为[0, 255]
cv.normalize(img, normalized_img, alpha=20, beta=235, norm_type=cv.NORM_MINMAX)

# 显示归一化后的图像
cv.imshow("Normalized Image", normalized_img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite("L2409050001.009.normalized20-235.png", normalized_img)

# hist,bins = np.histogram(normalized_img.flatten(),256,[0,256])

# cdf = hist.cumsum()
# cdf_normalized = cdf * float(hist.max()) / cdf.max()

# plt.plot(cdf_normalized, color = 'b')
# plt.hist(normalized_img.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.show()
