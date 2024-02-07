# -*- coding: utf-8 -*-
"""
Module of generating training data for  classifier and segmentation model from karyotype chart.

Usage:
    - Import this module using `import mymodule`.
    - Use the functions provided by this module as needed.

Author: Lei Wang
Date: Jan 19, 2024
"""
import os
import sys
from utils.chromo_cv_utils import cv_imread

__author__ = "王磊"
__copyright__ = "Copyright 2023 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"


if __name__ == "__main__":
    # 检查命令行参数大于1
    if len(sys.argv) < 2:
        print("Usage: get_properties_of_pic.py <pic_path>")
        sys.exit(-1)

    # 获取命令行参数
    fp = sys.argv[1]

    # 检查目录是否存在
    if not os.path.exists(fp):
        print(f"Error: {fp} does not exist!")
        sys.exit(-1)

    img = cv_imread(fp)
    # 打印图片的所有属性
    print("Image properties:")
    print(f"    - shape: {img.shape}")
    print(f"    - size: {img.size}")
    print(f"    - dtype: {img.dtype}")
