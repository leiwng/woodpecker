# -*- coding: utf-8 -*-
"""
Module for evaluating AI segmentation and classification result.
1. Use three process to evaluate AI result:
    a. Original AI chromosome image use SIFT to match with karyotype chromosome image.
    b. Use CLAHE to enhance AI chromosome image, then use SIFT to match with karyotype chromosome image.
    c. Use cv2.matchShapes to match AI chromosome contour with karyotype chromosome contour.
2. The AI is considered correct whenever the result of any of the above three processes indicates that the AI is correct.

Usage:
    - Import this module using `import mymodule`.
    - Use the functions provided by this module as needed.

Author: Lei Wang
Date: Feb 23, 2024
"""


__author__ = "王磊"
__copyright__ = "Copyright 2023 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"


import os

from karyotype import Karyotype
from utils.chromo_cv_utils import (
    cv_imread,
    cv_imwrite,
    contour_bbox_img,
)


KYT_DIR = (
    r"E:\染色体测试数据\240227-zzlAI结果评估测试集选取圈状染色体测试骨架算法\TEST_PIC"
)
OUTPUT_DIR = r"E:\染色体测试数据\240227-zzlAI结果评估测试集选取圈状染色体测试骨架算法\OUTPUT_CHROMO"

for kyt_fn in os.listdir(KYT_DIR):
    kyt_basename, kyt_ext = os.path.splitext(kyt_fn)
    output_dir_fp = os.path.join(OUTPUT_DIR, kyt_basename)
    if not os.path.exists(output_dir_fp):
        os.makedirs(output_dir_fp)

    kyt_fp = os.path.join(KYT_DIR, kyt_fn)
    kyt_img = cv_imread(kyt_fp)

    kyt_chart = Karyotype(kyt_fp)
    kyt_chromos_orgby_cy = kyt_chart.read_karyotype()

    for kyt_chromos_in_row in kyt_chromos_orgby_cy.values():
        for kyt_chromo in kyt_chromos_in_row:
            kyt_chromo_bbg_bbox, kyt_chromo_wbg_bbox = contour_bbox_img(
                kyt_img, kyt_chromo["cntr"]
            )
            # black background bbox
            output_fn = (
                f"chromo_{kyt_chromo['chromo_id']}-{kyt_chromo['cx']}_bbg-bbox.png"
            )
            output_fp = os.path.join(output_dir_fp, output_fn)
            cv_imwrite(output_fp, kyt_chromo_bbg_bbox)
            # white background bbox
            output_fn = (
                f"chromo_{kyt_chromo['chromo_id']}-{kyt_chromo['cx']}_wbg-bbox.png"
            )
            output_fp = os.path.join(output_dir_fp, output_fn)
            cv_imwrite(output_fp, kyt_chromo_wbg_bbox)
            print(f"Saved: {output_fp}")
