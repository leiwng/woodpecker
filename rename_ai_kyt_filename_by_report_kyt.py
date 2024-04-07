# -*- coding: utf-8 -*-
"""
对AI推理结果的目录名进行更名，使之能与核型报告图的文件名对应上

Usage:
    - Import this module using `import mymodule`.
    - Use the functions provided by this module as needed.

Author: Lei Wang
Date: Feb 4, 2024
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


KYT_IMG_DIR = r"E:\染色体测试数据\240204-评估240202测试集AI推理结果\KYT_IMG"
AI_KYT_IMG_DIR = r"E:\染色体测试数据\240204-评估240202测试集AI推理结果\AI_KYT"

# 缺的图
# missed_case_pic_id = ["A2402020001.051", "A2402020001.135"]

if not os.path.exists(AI_KYT_IMG_DIR):
    print(f"AI结果目录不存在: {AI_KYT_IMG_DIR}")
    sys.exit(1)
# 遍历AI结果目录，得到AI结果目录名列表
ai_kyt_img_fns = os.listdir(AI_KYT_IMG_DIR)
# 对该列表按字母顺序排序
ai_kyt_img_fns.sort()
ai_result_dirs_len = len(ai_kyt_img_fns)

if not os.path.exists(KYT_IMG_DIR):
    print(f"核型报告图目录不存在: {KYT_IMG_DIR}")
    sys.exit(1)
# 遍历核型报告图目录，得到核型报告图文件名列表
kyt_img_fns = os.listdir(KYT_IMG_DIR)
# 对该列表按字母顺序排序
kyt_img_fns.sort()
kyt_img_files_len = len(kyt_img_fns)

for ai_kyt_img_fn, kyt_img_fn in zip(ai_kyt_img_fns, kyt_img_fns):
    if ai_kyt_img_fn == ".".join(kyt_img_fn.split(".")[:2]):
        continue
    src_fp = os.path.join(AI_KYT_IMG_DIR, ai_kyt_img_fn)
    dst_fn = f"{'.'.join(kyt_img_fn.split('.')[:2])}.AIK-{'.'.join(ai_kyt_img_fn.split('.')[:2])}.png"
    dst_fp = os.path.join(AI_KYT_IMG_DIR, dst_fn)
    os.rename(src_fp, dst_fp)
