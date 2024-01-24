# -*- coding: utf-8 -*-
"""尝试多种不同的二值化方法以期达到较好的泛化性效果

Returns:
    _type_: _description_
"""

import os
import sys
import cv2
import numpy as np


def cv_imread(file_path):
    """读取带中文路径的图片文件
    Args:
        file_path (_type_): _description_
    Returns:
        _type_: _description_
    """
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)


def cv_imwrite(file_path, img):
    """保存带中文路径的图片文件
    Args:
        file_path (_type_): _description_
        img (_type_): _description_
    """
    cv2.imencode(".png", img)[1].tofile(file_path)


def bin_thresh_otsu(gray):
    """Otsu二值化
    Args:
        gray (_type_): _description_
    Returns:
        _type_: _description_
    """
    best_thresh, bin_img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return best_thresh, bin_img


def bin_thresh_triangle(gray):
    """Triangle二值化
    Args:
        gray (_type_): _description_
    Returns:
        _type_: _description_
    """
    best_thresh, bin_img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )
    return best_thresh, bin_img


def bin_thresh_adaptive_mean(gray, block_size=11, c=2):
    """自适应均值二值化
    Args:
        gray (_type_): _description_
    Returns:
        _type_: _description_
    """
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
    )


def bin_thresh_adaptive_gaussian(gray, block_size=11, c=2):
    """自适应高斯二值化
    Args:
        gray (_type_): _description_
    Returns:
        _type_: _description_
    """
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


if __name__ == "__main__":
    # 检查命令行参数大于1
    if len(sys.argv) < 2:
        print("Usage: bmp_cvt_to_png.py <dir_path>")
        sys.exit(0)

    # 获取命令行参数
    dir_fp = sys.argv[1]

    # 检查是否是目录
    if not os.path.isdir(dir_fp):
        print(f"Error: {dir_fp} is not a directory!")
        sys.exit(0)

    # 检查目录是否存在
    if not os.path.exists(dir_fp):
        print(f"Error: {dir_fp} does not exist!")
        sys.exit(0)

    for entry in os.listdir(dir_fp):
        # 如果不是图片文件则跳过
        if os.path.splitext(entry)[1].lower() not in [".png", ".jpg", ".bmp"]:
            continue

        full_path = os.path.join(dir_fp, entry)
        img = cv_imread(full_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # bin_thresh_otsu
        best_thresh, bin_img = bin_thresh_otsu(gray)
        new_fn = f"{os.path.splitext(full_path)[0]}_otsu.png"
        cv_imwrite(new_fn, bin_img)

        # bin_thresh_triangle
        best_thresh, bin_img = bin_thresh_triangle(gray)
        new_fn = f"{os.path.splitext(full_path)[0]}_triangle.png"
        cv_imwrite(new_fn, bin_img)

        # bin_thresh_adaptive_mean
        bin_img = bin_thresh_adaptive_mean(gray)
        new_fn = f"{os.path.splitext(full_path)[0]}_apt_mean.png"
        cv_imwrite(new_fn, bin_img)

        # bin_thresh_adaptive_gaussian
        bin_img = bin_thresh_adaptive_gaussian(gray)
        new_fn = f"{os.path.splitext(full_path)[0]}_apt_gaussian.png"
        cv_imwrite(new_fn, bin_img)
