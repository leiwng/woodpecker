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
import cv2
from karyotype import Karyotype
from utils.chromo_cv_utils import (
    cv_imread,
    cv_imwrite,
)


# KYT_ROOT_DIR = "E:\\染色体测试数据\\240129-用标准的报告图测试karyotpye.py"

KYT_ROOT_DIR = r"E:\染色体测试数据\240318-ylf-报核型图解析出错-第4排染色体编号数量为5应该为6-出错图片"

for dir_fn in os.listdir(KYT_ROOT_DIR):
    # 如果是文件则跳过
    if os.path.isfile(os.path.join(KYT_ROOT_DIR, dir_fn)):
        continue

    # 下面只处理目录
    for fn in os.listdir(os.path.join(KYT_ROOT_DIR, dir_fn)):
        # 如果不是文件则跳过
        if not os.path.isfile(os.path.join(KYT_ROOT_DIR, dir_fn, fn)):
            continue

        # 如果文件夹名不是tif,bmp或jpg，则跳过
        if os.path.splitext(fn)[1].lower() not in [".tif", ".jpg", ".bmp"]:
            continue

        # 构造核型图对象
        karyotype_chart = Karyotype(os.path.join(KYT_ROOT_DIR, dir_fn, fn))
        # 读取核型图
        karyotype_chart.read_karyotype()

        # 构造画布
        canvas = cv_imread(os.path.join(KYT_ROOT_DIR, dir_fn, fn))

        # 绘制染色体编号轮廓
        for cntrs in karyotype_chart.id_cntr_dicts_orgby_cy.values():
            for cntr in cntrs:
                cv2.drawContours(canvas, [cntr["cntr"]], -1, (255, 0, 0), 3)

        # 绘制染色体轮廓
        for cntrs in karyotype_chart.chromo_cntr_dicts_orgby_cy.values():
            for idx, cntr in enumerate(cntrs):
                cv2.drawContours(canvas, [cntr["cntr"]], -1, (0, 255, 0), 1)
                x = cntr["cx"]
                y = cntr["cy"]
                if idx % 2 == 0:
                    cv2.putText(
                        canvas,
                        cntr["chromo_id"],
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        canvas,
                        cntr["chromo_id"],
                        (x - 8, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        2,
                    )

        # 保存画布
        base_name = os.path.splitext(fn)[0]
        cv_imwrite(os.path.join(KYT_ROOT_DIR, dir_fn, f"{base_name}_kyt.png"), canvas)
