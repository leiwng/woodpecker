# -*- coding: utf-8 -*-
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


def convert_to_png(dir_path):
    """遍历目录将所有的bmp文件转换为png文件

    Args:
        dir_path (_type_): _description_
    """
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if os.path.isdir(full_path):
            print(f"Entering {full_path} ...")
            convert_to_png(full_path)  # Recursive call
        else:
            # chk if file is a bmp file
            if os.path.splitext(full_path)[1].lower() != ".bmp":
                continue
            print(f"Converting {full_path} ...")
            # read image file
            img = cv_imread(full_path)
            # convert to png
            new_fn = f"{os.path.splitext(full_path)[0]}.png"
            cv_imwrite(new_fn, img)
            # delete bmp file
            os.remove(full_path)


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

    # 检查文件是否存在
    if not os.path.exists(dir_fp):
        print(f"Error: {dir_fp} does not exist!")
        sys.exit(0)


    # 转换目录下的所有bmp文件为png文件
    convert_to_png(dir_fp)
