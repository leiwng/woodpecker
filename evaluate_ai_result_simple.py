# -*- coding: utf-8 -*-
"""
Module for simplified evaluating process on AI segmentation and classification result.

Usage:
    - Import this module using `import mymodule`.
    - Use the functions provided by this module as needed.

Author: Lei Wang
Date: Feb 19, 2024
"""


__author__ = "王磊"
__copyright__ = "Copyright 2023 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"


import os
from copy import deepcopy
import json
import time

from karyotype import Karyotype
from evaluate_ai_result_logger import Logger
from evaluate_ai_result_time_logger import TimeLogger


# 人工核型报告图图片目录
KYT_IMG_DIR = r"E:\染色体测试数据\240204-评估240202测试集AI推理结果\KYT_IMG"
# AI推理结果的根目录
AI_RESULT_ROOT_DIR = r"E:\染色体测试数据\240204-评估240202测试集AI推理结果\AI_RESULT"
# 保存评估结果保存的目录
EVA_RESULT_DIR = (
    r"E:\染色体测试数据\240204-评估240202测试集AI推理结果\EVA_RESULT_SIMPLE"
)

log = Logger.log()

if __name__ == "__main__":

    if not os.path.exists(KYT_IMG_DIR):
        log.info(f"核型报告图目录: {KYT_IMG_DIR} 不存在")

    if not os.path.exists(AI_RESULT_ROOT_DIR):
        log.info(f"AI结果目录: {AI_RESULT_ROOT_DIR} 不存在")

    if not os.path.exists(EVA_RESULT_DIR):
        os.makedirs(EVA_RESULT_DIR)

    # 初始化时间记录器
    case_pic_dirs = os.listdir(AI_RESULT_ROOT_DIR)
    case_pic_total = len(case_pic_dirs)
    t_log = TimeLogger(log, case_pic_total)

    # 所有图片的AI染色体个数记录
    all_pic_ai_chromo_cnt_dict = {}
    all_pic_kyt_chromo_cnt_dict = {}
    all_pic_chromo_cnt_diff_dict = {}
    all_kyt_chromo_cnt = 0
    all_ai_chromo_cnt = 0
    all_chromo_cnt_diff = 0

    # 根据AI的识别结果目录，同核型报告图进行比对，计算染色体的数量差异
    # 首先遍历AI识别结果目录，然后根据文件名找到对应的核型报告图，
    # 然后对报告图进行解析获得染色体信息，
    # 最后计算AI结果中每类染色体同报告图中同类染色体的数量差异
    for case_pic_dir in case_pic_dirs:
        # 获取AI结果的案例号和图号目录的full path
        case_pic_dir_fp = os.path.join(AI_RESULT_ROOT_DIR, case_pic_dir)

        # 应该都是案例和图号的目录，文件就跳过
        if os.path.isfile(case_pic_dir_fp):
            continue

        log.info("  ")
        log.info(f"    vvvvvvvvvv  开始处理新的AI结果:{case_pic_dir}  vvvvvvvvvv")
        log.info("  ")

        t_log.case_started(case_pic_dir)

        # 用于保存AI识别的染色体信息
        ai_chromo_cnt_dict = {}

        # 逐个获取AI结果
        for chromo_dir in os.listdir(case_pic_dir_fp):
            # 获取AI结果染色体图片目录的full path
            chromo_dir_fp = os.path.join(case_pic_dir_fp, chromo_dir)

            # 应该都是染色体编号的目录，文件就跳过
            if os.path.isfile(chromo_dir_fp):
                continue

            ai_chromo_cnt_dict[chromo_dir] = len(os.listdir(chromo_dir_fp))

        all_pic_ai_chromo_cnt_dict[case_pic_dir] = deepcopy(ai_chromo_cnt_dict)

        log.info(
            f"{case_pic_dir}: AI识别结果获取完毕。AI分割识别出的染色体共 {sum(ai_chromo_cnt_dict.values())} 条。"
        )

        # 获取对应案例和图号的核型报告图
        kyt_img_fn = f"{case_pic_dir}.K.JPG"
        kyt_img_fp = os.path.join(KYT_IMG_DIR, kyt_img_fn)
        # 读取报告图中染色体的信息
        kyt_chart = Karyotype(kyt_img_fp)
        kyt_chromo_cntr_dicts_orgby_cy = kyt_chart.read_karyotype()

        # 简化核型报告图中的染色体信息的组织结构，并计算bbox
        kyt_chromo_result = []
        for chromo_cntr_dicts in kyt_chromo_cntr_dicts_orgby_cy.values():
            kyt_chromo_result.extend(iter(chromo_cntr_dicts))
        # 按染色体编号和cx排序
        kyt_chromo_result = sorted(
            kyt_chromo_result, key=lambda x: (x["chromo_idx"], x["cx"])
        )

        # 染色体条数评估结果
        chromo_cnt_diff_dict = {}

        # 逐个遍历核型报告图中的染色体去同AI识别结果中的染色体数量做比较
        kyt_chromo_cnt_dict = {}
        for kyt_chromo in kyt_chromo_result:
            chromo_idx_str = str(kyt_chromo["chromo_idx"])
            kyt_chromo_cnt_dict[chromo_idx_str] = (
                kyt_chromo_cnt_dict.get(chromo_idx_str, 0) + 1
            )

        all_pic_kyt_chromo_cnt_dict[case_pic_dir] = deepcopy(kyt_chromo_cnt_dict)

        for key, item in kyt_chromo_cnt_dict.items():
            ai_item = ai_chromo_cnt_dict.get(key, 0)
            # print(f"key: {key}, item: {item}, ai_item: {ai_item}")
            chromo_cnt_diff_dict[key] = {
                "kyt_chromo_cnt": item,
                "ai_chromo_cnt": ai_item,
                "chromo_cnt_diff": abs(ai_item - item),
            }

        total_kyt_chromo_cnt = sum(kyt_chromo_cnt_dict.values())
        total_ai_chromo_cnt = sum(ai_chromo_cnt_dict.values())
        total_chromo_cnt_diff = sum(
            chromo_cnt_diff_dict[key]["chromo_cnt_diff"]
            for key in chromo_cnt_diff_dict.keys()
        )
        total_chromo_cnt_diff_ratio = total_chromo_cnt_diff / total_kyt_chromo_cnt
        chromo_cnt_diff_dict["total_kyt_chromo_cnt"] = total_kyt_chromo_cnt
        chromo_cnt_diff_dict["total_ai_chromo_cnt"] = total_ai_chromo_cnt
        chromo_cnt_diff_dict["total_cnt_diff"] = total_chromo_cnt_diff
        chromo_cnt_diff_dict["total_cnt_diff_ratio"] = total_chromo_cnt_diff_ratio
        all_pic_chromo_cnt_diff_dict[case_pic_dir] = deepcopy(chromo_cnt_diff_dict)

        all_kyt_chromo_cnt += total_kyt_chromo_cnt
        all_ai_chromo_cnt += total_ai_chromo_cnt
        all_chromo_cnt_diff += total_chromo_cnt_diff

        t_log.case_finished(case_pic_dir)

    t_log.all_finished()

    # 保存评估结果
    # 保存评估结果的文件名为eva_result_开头后面接当前时间
    EVA_RESULT_FN = f"chromo_cnt_diff_result_{time.strftime('%Y%m%d%H%M%S')}.json"

    if not os.path.exists(EVA_RESULT_DIR):
        os.makedirs(EVA_RESULT_DIR)

    eva_result_fp = os.path.join(EVA_RESULT_DIR, EVA_RESULT_FN)

    # 结果保存为json文件
    with open(eva_result_fp, "w", encoding="utf-8") as f:
        json.dump(all_pic_chromo_cnt_diff_dict, f, ensure_ascii=False, indent=4)
        # 最后一行写入这批AI识别结果的计数差异
        all_chromo_cnt_diff_ratio = all_chromo_cnt_diff / all_kyt_chromo_cnt
        f.write(
            f"\n\n本批次({time.strftime('%Y%m%d%H%M%S')})染色体计数, 报告图染色体总数:{all_kyt_chromo_cnt}, AI推理染色体总数:{all_ai_chromo_cnt},差异数量: {all_chromo_cnt_diff}, 差异比例为: {all_chromo_cnt_diff_ratio}"
        )
