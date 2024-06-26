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


KYT_IMG_DIR = r"E:\染色体测试数据\240320-ly论文100张测试数据\KYT_IMG"
AI_RESULT_ROOT_DIR = r"E:\染色体测试数据\240320-ly论文100张测试数据\KYT_IMG"

# 缺的图
# missed_case_pic_id = ["A2402020001.051", "A2402020001.135"]

if not os.path.exists(AI_RESULT_ROOT_DIR):
    print(f"AI结果目录不存在: {AI_RESULT_ROOT_DIR}")
    sys.exit(1)
# 遍历AI结果目录，得到AI结果目录名列表
ai_result_dirs = os.listdir(AI_RESULT_ROOT_DIR)
# 对该列表按字母顺序排序
ai_result_dirs.sort()
ai_result_dirs_len = len(ai_result_dirs)

if not os.path.exists(KYT_IMG_DIR):
    print(f"核型报告图目录不存在: {KYT_IMG_DIR}")
    sys.exit(1)
# 遍历核型报告图目录，得到核型报告图文件名列表
kyt_img_files = os.listdir(KYT_IMG_DIR)
# 对该列表按字母顺序排序
kyt_img_files.sort()
kyt_img_files_len = len(kyt_img_files)

if ai_result_dirs_len != kyt_img_files_len:
    print(
        f"AI结果目录名数量({ai_result_dirs_len})与核型报告图文件名数量({kyt_img_files_len})不一致"
    )
    sys.exit(1)

for ai_result_dir, kyt_img_file in zip(ai_result_dirs, kyt_img_files):

    if ai_result_dir == ".".join(kyt_img_file.split(".")[:2]):
        continue

    # 根据AI结果目录名来重命名核型报告图文件名
    src_fp = os.path.join(KYT_IMG_DIR, kyt_img_file)
    dst_fn = ".".join(ai_result_dir.split(".")[:2]) + ".K.JPG"
    dst_fp = os.path.join(KYT_IMG_DIR, dst_fn)
    os.rename(src_fp, dst_fp)
