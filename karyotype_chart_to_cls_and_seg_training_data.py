# -*- coding: utf-8 -*-
__author__ = "王磊"
__copyright__ = "Copyright 2023 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"

"""
Module of generating training data for  classifier and segmentation model from karyotype chart.

Usage:
    - Import this module using `import mymodule`.
    - Use the functions provided by this module as needed.

Author: Lei Wang
Date: Jan 19, 2024
"""

import os
import cv2

from karyotype import Karyotype


KYT_IMG_DIR = r"D:\染色体测试数据\240118-给AI团队准备报告图数据用于迁移训练\01_input\hx_karyotype_chart_1000pics"
CLS_OUTPUT_DIR = r"D:\染色体测试数据\240118-给AI团队准备报告图数据用于迁移训练\11_output\hx_af_karyotype_chart_to_cls_training_data_240118"
SEG_OUTPUT_DIR = r"D:\染色体测试数据\240118-给AI团队准备报告图数据用于迁移训练\11_output\hx_af_karyotype_chart_to_seg_training_data_240118"


if os.path.exists(KYT_IMG_DIR) is False:
    print("Karyotype chart image directory does not exist!")
    exit(-1)

if os.path.exists(CLS_OUTPUT_DIR) is False:
    os.makedirs(CLS_OUTPUT_DIR)

if os.path.exists(SEG_OUTPUT_DIR) is False:
    os.makedirs(SEG_OUTPUT_DIR)

for img_fn in os.listdir(KYT_IMG_DIR):
    # 文件名被"."分割为4部分:案例号，图号，核型图标识，文件扩展名
    # 蔡司的核型图导出文件命名举例：L2311245727.001.K.TIF
    # 徕卡的核型图导出文件命名举例：A160029.0687.1.tif
    fn_splits = img_fn.split(".")
    # 如果文件名按"."分割后的列表长度不为4，则跳过该文件
    if len(fn_splits) != 4:
        continue
    # 如果文件扩展名不是tif和JPG，则跳过该文件
    if fn_splits[3].upper() not in ["TIF", "JPG"]:
        continue

    case_id = fn_splits[0]
    chart_id = fn_splits[1]
    chart_type = fn_splits[2]
    file_ext = fn_splits[3]

    karyotype = Karyotype(os.path.join(KYT_IMG_DIR, img_fn))
    chromo_cntr_dicts_orgby_cy = karyotype.read_karyotype()

    for cntrs in chromo_cntr_dicts_orgby_cy:
        for cntr in cntrs:
            # 生成分割模型训练数据
            # 掩码图像crop_picture分辨率为 1280x1024，二值化后的图像，背景为黑色，染色体为白色
            # 中期图mid_picture分辨率为 1280x1024

