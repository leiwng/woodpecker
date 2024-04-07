"""_summary_
"""

import cv2
from utils.chromo_cv_utils import cv_imread

IMG_FP = r"E:\染色体测试数据\240320-ly论文100张测试数据\ORI_IMG\A2308306219.090.PNG"

# 读取图像
image = cv_imread(IMG_FP)

# 初始化低阈值和高阈值
low_threshold = 50
high_threshold = 185

# 创建窗口和滑动条
cv2.namedWindow("image")
cv2.createTrackbar("Low Threshold", "image", low_threshold, 255, lambda x: None)
cv2.createTrackbar("High Threshold", "image", high_threshold, 255, lambda x: None)

while True:
    # 应用Canny边缘检测
    edges = cv2.Canny(
        image,
        cv2.getTrackbarPos("Low Threshold", "image"),
        cv2.getTrackbarPos("High Threshold", "image"),
    )

    # 显示边缘图像
    cv2.imshow("image", edges)

    # 等待按键
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
