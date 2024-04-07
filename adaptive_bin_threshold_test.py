"""
Module for getting chromosome image, mask and contour with chromosome id from karyotype image.

Usage:
    - Import this module using `import mymodule`.
    - Use the functions provided by this module as needed.

Author: Lei Wang
Date: Dec 14, 2023
"""

__author__ = "王磊"
__copyright__ = "Copyright 2023 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"


import os
import sys
import cv2
from utils.chromo_cv_utils import (
    cv_imread,
    cv_imwrite,
    find_external_contours_en,
)

ORI_IMG_DIR = r"E:\染色体测试数据\240320-ly论文100张测试数据\ORI_IMG"

# 检查原始图像目录是否存在
if not os.path.exists(ORI_IMG_DIR):
    print(f"原始图像目录不存在: {ORI_IMG_DIR}")
    sys.exit(1)

# 遍历原始图像目录，对每张原始图像进行求轮廓处理
for fn in os.listdir(ORI_IMG_DIR):
    # 如果不是文件则跳过
    if not os.path.isfile(os.path.join(ORI_IMG_DIR, fn)):
        continue

    # 如果文件夹名不是tif,bmp或jpg，则跳过
    if os.path.splitext(fn)[1].lower() not in [".tif", ".jpg", ".bmp", ".png"]:
        continue

    # 读取原始图像
    ori_img = cv_imread(os.path.join(ORI_IMG_DIR, fn))

    # 求轮廓
    external_contours, bin_thresh = find_external_contours_en(
        img=ori_img,
        bin_thresh=-1,
        bin_type=cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE,
        bin_thresh_adjustment=-50,
    )

    # 绘制轮廓
    cv2.drawContours(ori_img, external_contours, -1, (0, 0, 255), 1)

    # 保存绘制了轮廓的原始图像
    fbn = os.path.splitext(fn)[0]
    fext = os.path.splitext(fn)[1]

    cv_imwrite(os.path.join(ORI_IMG_DIR, f"{fbn}_cntr{fext}"), ori_img)

    # 打印最终的二值化阈值
    print(f"bin_thresh: {bin_thresh}")
