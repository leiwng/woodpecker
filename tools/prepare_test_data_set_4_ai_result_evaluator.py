# -*- coding: utf-8 -*-
""" 准备AI结果评估程序的测试数据集
    1. 根据核型报告图目录中的文件名，在AI识别结果目录中生成对应以文件名命名的空白目录
    2. 用测试数据填充AI识别结果目录中的空白目录
"""

import os
from shutil import copyfile


KYT_IMG_DIR = r"E:\染色体测试数据\240202-测试AI结果评估程序\KYT_IMG"
AI_RESULT_ROOT_DIR = r"E:\染色体测试数据\240202-测试AI结果评估程序\AI_RESULT"
TEST_DATA_SET_DIR = r"E:\染色体测试数据\240202-测试AI结果评估程序\test_data_set"


# 根据核型报告图目录中的文件名，在AI识别结果目录中生成对应以文件名命名的空白目录
for fn in os.listdir(KYT_IMG_DIR):
    if not os.path.isfile(os.path.join(KYT_IMG_DIR, fn)):
        continue

    fbasename = os.path.splitext(fn)[0]
    # 如果basename被"."分割后的列表长度不为3，则取前面的部分重新拼接成新的basename
    if len(fbasename.split(".")) >= 3:
        fbasename = ".".join(fbasename.split(".")[:2])

    if os.path.exists(os.path.join(AI_RESULT_ROOT_DIR, fbasename)) is False:
        os.makedirs(os.path.join(AI_RESULT_ROOT_DIR, fbasename))

# 用测试数据填充AI识别结果目录中的空白目录
for case_pic_idx, fn in enumerate(os.listdir(AI_RESULT_ROOT_DIR)):
    # 应该都是案例和图号的目录，文件就跳过
    if os.path.isfile(os.path.join(AI_RESULT_ROOT_DIR, fn)):
        continue

    case_pic_dir_fp = os.path.join(AI_RESULT_ROOT_DIR, fn)

    for chromo_id_idx in range(24):
        # 创建染色体图片目录
        chromo_pic_dir_fp = os.path.join(case_pic_dir_fp, f"{chromo_id_idx}")
        if os.path.exists(chromo_pic_dir_fp) is False:
            os.makedirs(chromo_pic_dir_fp)

        # 从测试数据集中复制测试数据到染色体图片目录
        src_chromo_dir = os.path.join(TEST_DATA_SET_DIR, f"{chromo_id_idx}")

        # 取同源第一根染色体的图片
        src_chromo_fn = case_pic_idx * 2
        src_chromo_fp = os.path.join(src_chromo_dir, f"{src_chromo_fn}.png")
        # 拷贝同源第一根染色体的图片到AI识别结果目录中的染色体图片目录
        dst_chromo_pic_fp = os.path.join(chromo_pic_dir_fp, "0.png")
        copyfile(src_chromo_fp, dst_chromo_pic_fp)

        # 取同源第二根染色体的图片
        src_chromo_fn = case_pic_idx * 2 + 1
        src_chromo_fp = os.path.join(src_chromo_dir, f"{src_chromo_fn}.png")
        # 拷贝同源第一根染色体的图片到AI识别结果目录中的染色体图片目录
        dst_chromo_pic_fp = os.path.join(chromo_pic_dir_fp, "1.png")
        copyfile(src_chromo_fp, dst_chromo_pic_fp)
