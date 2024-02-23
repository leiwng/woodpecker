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
import time
import json
import traceback
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Any
import cv2
import numpy as np
from karyotype import Karyotype
from utils.chromo_cv_utils import (
    cv_imread,
    cv_imwrite,
    find_external_contours,
    contour_bbox_img,
    sift_similarity_on_roi,
)
from evaluate_ai_result_logger import Logger
from evaluate_ai_result_time_logger import TimeLogger


# 人工核型报告图图片目录
KYT_IMG_DIR = r"E:\染色体测试数据\240204-评估240202测试集AI推理结果\KYT_IMG"
# AI推理结果的根目录
AI_RESULT_ROOT_DIR = r"E:\染色体测试数据\240204-评估240202测试集AI推理结果\AI_RESULT"
# 核型报告图解析的结果图片保存目录，用于调试
DBG_PIC_DIR = r"E:\染色体测试数据\240204-评估240202测试集AI推理结果\DBG_PIC"
# 保存评估结果保存的目录
EVA_RESULT_DIR = r"E:\染色体测试数据\240204-评估240202测试集AI推理结果\EVA_RESULT"

# 初始化日志
LogPath = Path(__file__).resolve().parent
# BasePath = os.path.dirname(os.path.abspath(__file__))
# LogPath = os.path.join(BasePath, "logs")
CurrentFileName = os.path.splitext(os.path.basename(__file__))[0]
LogFileName = f"{CurrentFileName}.log"
log = Logger.log(LogPath, LogFileName)

# KYT_IMG_DIR = r"E:\染色体测试数据\240202-测试AI结果评估程序\240206-bug_fix_4_sift_met_exp\KYT_IMG"
# AI_RESULT_ROOT_DIR = r"E:\染色体测试数据\240202-测试AI结果评估程序\240206-bug_fix_4_sift_met_exp\AI_RESULT"
# DBG_PIC_DIR = r"E:\染色体测试数据\240202-测试AI结果评估程序\240206-bug_fix_4_sift_met_exp\DBG_PIC"
# EVA_RESULT_DIR = r"E:\染色体测试数据\240202-测试AI结果评估程序\240206-bug_fix_4_sift_met_exp\EVA_RESULT"

if not os.path.exists(KYT_IMG_DIR):
    log.info(f"核型报告图目录: {KYT_IMG_DIR} 不存在")

if not os.path.exists(AI_RESULT_ROOT_DIR):
    log.info(f"AI结果目录: {AI_RESULT_ROOT_DIR} 不存在")

if not os.path.exists(DBG_PIC_DIR):
    os.makedirs(DBG_PIC_DIR)

if not os.path.exists(EVA_RESULT_DIR):
    os.makedirs(EVA_RESULT_DIR)

# 初始化时间记录器
case_pic_dirs = os.listdir(AI_RESULT_ROOT_DIR)
case_pic_total = len(case_pic_dirs)
t_log = TimeLogger(log, case_pic_total)

# 根据AI的识别结果目录，同核型报告图进行比对，计算准确率
# 首先遍历AI识别结果目录，然后根据文件名找到对应的核型报告图，
# 然后对报告图进行解析获得染色体信息，
# 最后对比AI识别结果和核型报告图的染色体信息，进行匹配，计算准确率
for case_pic_dir in case_pic_dirs:
    # 获取AI结果的案例号和图号目录的full path
    case_pic_dir_fp = os.path.join(AI_RESULT_ROOT_DIR, case_pic_dir)

    # 应该都是案例和图号的目录，文件就跳过
    if os.path.isfile(case_pic_dir_fp):
        continue

    log.info("  ")
    log.info(f"    vvvvvvvvvv  开始处理新CASE: {case_pic_dir}  vvvvvvvvvv")
    log.info("  ")

    t_log.case_started(case_pic_dir)

    # 初始化该案例和图号的评估结果的存储结构
    eva_result_dict = {"case_pic_id": case_pic_dir}

    # 用于保存AI识别的染色体信息
    ai_chromo_result = []
    # 将AI结果按染色体序号chromo_idx进行组织
    ChromoData = Dict[str, Any]
    ai_chromo_orgby_chromo_idx: Dict[int, List[ChromoData]] = {}

    # 逐个获取AI结果
    log.info("提取AI推理结果")
    for chromo_dir in os.listdir(case_pic_dir_fp):
        # 获取AI结果染色体图片目录的full path
        chromo_dir_fp = os.path.join(case_pic_dir_fp, chromo_dir)

        # 应该都是染色体编号的目录，文件就跳过
        if os.path.isfile(chromo_dir_fp):
            continue

        for chromo_pic_fn in os.listdir(chromo_dir_fp):
            # 获取染色体图片文件的full path
            chromo_pic_fp = os.path.join(chromo_dir_fp, chromo_pic_fn)

            # 应该都是染色体图片文件，不是就跳过
            if not os.path.isfile(chromo_pic_fp):
                continue

            # 获取AI识别的染色体信息
            chromo_img = cv_imread(chromo_pic_fp)
            if chromo_img is None:
                raise ValueError(f"{chromo_pic_fp} is not a valid image")

            chromo_idx = int(chromo_dir)
            if chromo_idx == 22:
                chromo_id = "X" # pylint: disable=invalid-name
            elif chromo_idx == 23:
                chromo_id = "Y" # pylint: disable=invalid-name
            else:
                chromo_id = str(chromo_idx + 1) # pylint: disable=invalid-name

            chromo_cntr = find_external_contours(chromo_img, 253)[0]
            grayscale = cv2.cvtColor(chromo_img, cv2.COLOR_BGR2GRAY)
            chromo_mask = np.zeros_like(grayscale, dtype=np.uint8)
            cv2.drawContours(chromo_mask, [chromo_cntr], -1, 255, thickness=cv2.FILLED)
            chromo_roi = cv2.bitwise_and(grayscale, grayscale, mask=chromo_mask)
            chromo_bbox_bbg, chromo_bbox_wbg = contour_bbox_img(chromo_img, chromo_cntr)
            chromo_info_dict = {
                    "img": chromo_img,
                    "idx": chromo_idx,
                    "id": chromo_id,
                    "cntr": find_external_contours(chromo_img, 253)[0],
                    "roi": chromo_roi,
                    "position": int(os.path.splitext(chromo_pic_fn)[0]),
                    "bbox_bbg": chromo_bbox_bbg,
                    "bbox_wbg": chromo_bbox_wbg,
            }
            # 保存到相关的数据结构中
            ai_chromo_result.append(deepcopy(chromo_info_dict))
            if chromo_idx not in ai_chromo_orgby_chromo_idx:
                ai_chromo_orgby_chromo_idx[chromo_idx] = []
            ai_chromo_orgby_chromo_idx[chromo_idx].append(deepcopy(chromo_info_dict))

    # 按染色体编号和cx排序
    ai_chromo_result = sorted(ai_chromo_result, key=lambda x: (x['idx'], x['position']))
    for chromos in ai_chromo_orgby_chromo_idx.values():
        chromos.sort(key=lambda x: x['position'])

    log.info(f"AI推理结果获取完毕。AI分割识别出的染色体共 {len(ai_chromo_result)} 条。")

    log.info("开始解析核型报告图中的染色体信息")
    
    # 获取对应案例和图号的核型报告图
    kyt_img_fn = f"{case_pic_dir}.K.JPG"
    kyt_img_fp = os.path.join(KYT_IMG_DIR, kyt_img_fn)
    # 读取报告图中染色体的信息
    kyt_chart = Karyotype(kyt_img_fp)
    kyt_chromo_cntr_dicts_orgby_cy = kyt_chart.read_karyotype()

    # 简化核型报告图中的染色体信息的组织结构，并计算bbox
    kyt_chromo_result = []
    kyt_chromo_orgby_chromo_idx: Dict[int, List[ChromoData]] = {}
    for chromo_cntr_dicts in kyt_chromo_cntr_dicts_orgby_cy.values():
        for chromo_cntr_dict in chromo_cntr_dicts:
            chromo_bbox_bbg, chromo_bbox_wbg = contour_bbox_img(
                kyt_chart.img["img"], chromo_cntr_dict["cntr"]
            )
            chromo_cntr_dict["bbox_bbg"] = chromo_bbox_bbg
            chromo_cntr_dict["bbox_wbg"] = chromo_bbox_wbg
            kyt_chromo_result.append(chromo_cntr_dict)
            chromo_idx = chromo_cntr_dict["chromo_idx"]
            if chromo_idx not in kyt_chromo_orgby_chromo_idx:
                kyt_chromo_orgby_chromo_idx[chromo_idx] = []
            kyt_chromo_orgby_chromo_idx[chromo_idx].append(chromo_cntr_dict)

    # 按染色体编号和cx排序
    kyt_chromo_result = sorted(kyt_chromo_result, key=lambda x: (x['chromo_idx'], x['cx']))
    # 将染色体的cx坐标转换为position,最左边的染色体为0, 依次递增
    cur_chromo_idx = 0 # pylint: disable=invalid-name
    pre_chromo_idx = 0 # pylint: disable=invalid-name
    pos = 0 # pylint: disable=invalid-name
    for chromo in kyt_chromo_result:
        cur_chromo_idx = chromo["chromo_idx"]
        if cur_chromo_idx != pre_chromo_idx:
            pre_chromo_idx = cur_chromo_idx
            pos = 0 # pylint: disable=invalid-name
        chromo["position"] = pos
        pos += 1
    for chromos in kyt_chromo_orgby_chromo_idx.values():
        chromos.sort(key=lambda x: x['cx'])
        for i, chromo in enumerate(chromos):
            chromo["position"] = i

    # 打印核型报告图中的染色体信息图片用于调试
    # canvas = kyt_chart.img["img"].copy()
    # for chromo_cntr_dict in kyt_chromo_result:
    #     cv2.drawContours(canvas, [chromo_cntr_dict["cntr"]], -1, (0, 0, 255), 1)
    #     x = chromo_cntr_dict["cx"]
    #     y = chromo_cntr_dict["cy"]
    #     chromo_id = chromo_cntr_dict["chromo_id"]
    #     cv2.putText(
    #         canvas, chromo_id, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2
    #     )

    # # 保存核型报告图中的染色体信息图片用于调试
    # dbg_pic_fn = f"{case_pic_dir}.K.kyt_parse_dbg.JPG"
    # dbg_pic_fp = os.path.join(DBG_PIC_DIR, dbg_pic_fn)
    # cv_imwrite(dbg_pic_fp, canvas)

    log.info(f"核型报告图解析完毕。核型报告图中的染色体共 {len(kyt_chromo_result)} 条。")

    # 对比核型报告图中的，和AI识别结果中的染色体信息，进行匹配，计算准确率
    # 染色体编号一致的认为AI推理结果正确
    SAME_CNT = 0

    # 逐个遍历核型报告图中的染色体去找到对应的AI识别的染色体
    for kyt_chromo in kyt_chromo_result:
        MAX_SIM = 0
        AI_CHROMO_ON_MAX = None
        KYT_CHROMO_ON_MAX = None
        UP_SIDE_DOWN = False

        # 每一个kyt_chromo试图去找到对应的ai_chromo
        # 如果kyt_chromo是3号染色体，而ai_chromo中没有3号,
        # 那么必定会产生错误匹配,这样就仍然对准确率有贡献,不会漏掉.
        MET_EXP = False
        for ai_chromo in ai_chromo_result:
            # 不翻转，原图计算相似度
            FINISH_STEP = "sim0"
            try:
                sim1 = sift_similarity_on_roi(
                    ai_chromo["bbox_bbg"], kyt_chromo["bbox_bbg"]
                )
                FINISH_STEP = "sim1"

                # 水平翻转(flip参数送1)，计算相似度
                sim2 = sift_similarity_on_roi(
                    cv2.flip(ai_chromo["bbox_bbg"], 1), kyt_chromo["bbox_bbg"]
                )
                FINISH_STEP = "sim2"

                # 垂直翻转(flip参数送0)，计算相似度
                sim3 = sift_similarity_on_roi(
                    cv2.flip(ai_chromo["bbox_bbg"], 0), kyt_chromo["bbox_bbg"]
                )
                FINISH_STEP = "sim3"

                # 水平垂直翻转(flip参数送-1)，计算相似度
                sim4 = sift_similarity_on_roi(
                    cv2.flip(ai_chromo["bbox_bbg"], -1), kyt_chromo["bbox_bbg"]
                )
                FINISH_STEP = "sim4"

            except Exception as e:
                MET_EXP = True
                kyt_id = kyt_chromo["chromo_id"]
                ai_id = ai_chromo["id"]
                log.error(
                    f"Case: {case_pic_dir}, On step:{FINISH_STEP} using KYT Chromo: {kyt_id} match AI Chromo: {ai_id} met Error: {e}"
                )
                # 默认在终端打印异常, 默认颜色为红色
                traceback.print_exc()
                # 接收错误信息
                err = traceback.format_exc()
                print(err)
                log.error(err)

                # 出现异常，跳过当前这个AI识别的染色体，继续下一个
                continue
                # raise e

            # end of try-except

            # 选择最大相似度
            sim = max(sim1, sim2, sim3, sim4)
            if sim > MAX_SIM:
                MAX_SIM = sim
                AI_CHROMO_ON_MAX = ai_chromo
                KYT_CHROMO_ON_MAX = kyt_chromo
                UP_SIDE_DOWN = sim3 > sim1 or sim4 > sim1

        if MET_EXP:
            # 保存报告图中的染色体图片用于调试
            dbg_pic_fn = (
                f"{case_pic_dir}.K.sim-exp-kytID{kyt_chromo["chromo_id"]}_{kyt_chromo["cx"]}.PNG"
            )
            dbg_pic_fp = os.path.join(DBG_PIC_DIR, dbg_pic_fn)
            cv_imwrite(dbg_pic_fp, kyt_chromo["bbox_bbg"])
            # 出现异常，跳过当前这个核型报告图中的染色体，继续下一个
            continue

        try:
            log.info(
                f"报告图中的染色体: {KYT_CHROMO_ON_MAX['chromo_id']}-{KYT_CHROMO_ON_MAX["cx"]} 同 AI识别的染色体: {AI_CHROMO_ON_MAX['id']}-{AI_CHROMO_ON_MAX["position"]} 最匹配, 相似度: {MAX_SIM:.2f} , 颠倒? {UP_SIDE_DOWN}。"
            )

            # 同当前AI识别的染色体相似度最高的核型报告图中的染色体已经找到
            eva_result_key = f"{KYT_CHROMO_ON_MAX['chromo_id']}-{KYT_CHROMO_ON_MAX["cx"]}"
            eva_result_dict[eva_result_key] = {
                "kyt_chromo_id": KYT_CHROMO_ON_MAX["chromo_id"],
                "kyt_chromo_cx": KYT_CHROMO_ON_MAX["cx"],
                "ai_chromo_id": AI_CHROMO_ON_MAX["id"],
                "ai_chromo_position_idx": AI_CHROMO_ON_MAX["position"],
                "similarity": MAX_SIM,
                "UP_SIDE_DOWN": UP_SIDE_DOWN,
            }

            if AI_CHROMO_ON_MAX["id"] == KYT_CHROMO_ON_MAX["chromo_id"] and not UP_SIDE_DOWN:
                SAME_CNT += 1

        except Exception as e:
            log.error(
                f"Case: {case_pic_dir}, On step: Summarize KYT Chromo: final match which AI Chromo met Error: {e}"
            )
            # 默认在终端打印异常, 默认颜色为红色
            traceback.print_exc()
            # 接收错误信息
            err = traceback.format_exc()
            print(err)
            log.error(err)
            # raise e

    # 所有AI识别的染色体都已经同核型报告图中的染色体匹配完毕
    eva_result_dict["acc_ratio"] = SAME_CNT / len(kyt_chromo_result)
    ALL_CASE_PIC_ACC_RATIO_SUM += eva_result_dict["acc_ratio"]

    # 保存当前案例下该图的评估结果
    eva_result.append(eva_result_dict)

    # 记录当前case+pic的时间
    t_log.case_finished(case_pic_dir)

    log.info(f"{case_pic_dir}处理完毕, 准确率: {eva_result_dict['acc_ratio']:.2f}")
    log.info(" ")
    log.info("    ^^^^^^^^^^  处理完毕  ^^^^^^^^^^")
    log.info(" ")

# 所有case+pic处理完毕记录总处理时间
t_log.all_finished()

# 所有案例下的所有报告图跑完了，计算平均准确率
acc_ratio_avg = ALL_CASE_PIC_ACC_RATIO_SUM / len(eva_result)
log.info(f"所有案例下的报告图评估完毕。AI推理的平均准确率为 {acc_ratio_avg:.2f} 。")

# 保存评估结果
# 保存评估结果的文件名为eva_result_开头后面接当前时间
EVA_RESULT_FN = f"evaluate_ai_result_{time.strftime('%Y%m%d%H%M%S')}.json"

if not os.path.exists(EVA_RESULT_DIR):
    os.makedirs(EVA_RESULT_DIR)

eva_result_fp = os.path.join(EVA_RESULT_DIR, EVA_RESULT_FN)
# 结果保存为json文件
with open(eva_result_fp, "w", encoding="utf-8") as f:
    json.dump(eva_result, f, ensure_ascii=False, indent=4)
    # 最后一行写入这批AI识别结果的平均准确率
    f.write(f"\n\n本批次({time.strftime('%Y%m%d%H%M%S')})平均准确率: {acc_ratio_avg}")

log.info(f"评估结果保存完毕。文件路径: {eva_result_fp}")
