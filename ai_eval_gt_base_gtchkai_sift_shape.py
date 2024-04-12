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
import traceback
from copy import deepcopy
from typing import Dict, List, Any
import cv2
import numpy as np
import pandas as pd

from utils.chromo_cv_utils import (
    cv_imread,
    # cv_imwrite,
    find_external_contours,
    contour_bbox_img,
    # sift_similarity_on_roi,
    best_shape_match_for_chromos,
    feature_match_on_roi_for_flips,
    best_feature_match_for_chromos,
)
from ai_eval_logger import Logger
from ai_eval_time_logger import TimeLogger


def get_chromo_info_from_result_dir(result_dir_fp: str):
    """从标注结果文件中获取染色体信息
    :param label_result_fp: 标注结果文件的路径
    :return: 染色体信息列表
    """

    # 用于保存AI识别的染色体信息
    ai_chromo_result = []
    # 将AI结果按染色体序号chromo_idx进行组织
    ai_chromo_orgby_chromo_idx: Dict[int, List[Dict[str, Any]]] = {}

    # 逐个获取AI结果
    for chromo_dir in os.listdir(result_dir_fp):
        # 获取AI结果染色体图片目录的full path
        chromo_dir_fp = os.path.join(result_dir_fp, chromo_dir)

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
                chromo_id = "X"  # pylint: disable=invalid-name
            elif chromo_idx == 23:
                chromo_id = "Y"  # pylint: disable=invalid-name
            else:
                chromo_id = str(chromo_idx + 1)  # pylint: disable=invalid-name

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
                "pos": int(os.path.splitext(chromo_pic_fn)[0]),
                "bbox_bbg": chromo_bbox_bbg,
                "bbox_wbg": chromo_bbox_wbg,
            }
            # 保存到相关的数据结构中
            ai_chromo_result.append(deepcopy(chromo_info_dict))
            if chromo_idx not in ai_chromo_orgby_chromo_idx:
                ai_chromo_orgby_chromo_idx[chromo_idx] = []
            ai_chromo_orgby_chromo_idx[chromo_idx].append(deepcopy(chromo_info_dict))

    # 按染色体编号和cx排序
    ai_chromo_result = sorted(ai_chromo_result, key=lambda x: (x["idx"], x["pos"]))
    for chromos in ai_chromo_orgby_chromo_idx.values():
        chromos.sort(key=lambda x: x["pos"])

    return ai_chromo_result, ai_chromo_orgby_chromo_idx


if __name__ == "__main__":

    # 标准答案
    GROUND_TRUTH_ROOT_DIR = r"E:\染色体测试数据\240407-评估AI准确性_绵阳妇幼在标注系统中L2403090001.001_L2403120008.040_完成_共890张图\GROUND_TRUTH"
    # AI推理结果的根目录
    AI_RESULT_ROOT_DIR = r"E:\染色体测试数据\240407-评估AI准确性_绵阳妇幼在标注系统中L2403090001.001_L2403120008.040_完成_共890张图\AI_RESULT"
    # 保存评估结果保存的目录
    EVA_RESULT_DIR = r"E:\染色体测试数据\240407-评估AI准确性_绵阳妇幼在标注系统中L2403090001.001_L2403120008.040_完成_共890张图\EVA_RESULT"

    # # 测试
    # # 标准答案
    # GROUND_TRUTH_ROOT_DIR = r"E:\染色体测试数据\tmp\240410-快速测试\GROUND_TRUTH"
    # # AI推理结果的根目录
    # AI_RESULT_ROOT_DIR = r"E:\染色体测试数据\tmp\240410-快速测试\AI_RESULT"
    # # 保存评估结果保存的目录
    # EVA_RESULT_DIR = r"E:\染色体测试数据\tmp\240410-快速测试\EVA_RESULT"

    # 初始化日志
    # LogPath = Path(__file__).resolve().parent
    LogPath = EVA_RESULT_DIR  # pylint: disable=invalid-name
    # BasePath = os.path.dirname(os.path.abspath(__file__))
    # LogPath = os.path.join(BasePath, "logs")
    CurrentFileName = os.path.splitext(os.path.basename(__file__))[0]
    LogFileName = f"{CurrentFileName}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log = Logger.log(LogPath, LogFileName)

    if not os.path.exists(GROUND_TRUTH_ROOT_DIR):
        log.info(f"标准答案(GROUND TRUTH)目录: {GROUND_TRUTH_ROOT_DIR} 不存在")

    if not os.path.exists(AI_RESULT_ROOT_DIR):
        log.info(f"AI结果目录: {AI_RESULT_ROOT_DIR} 不存在")

    if not os.path.exists(EVA_RESULT_DIR):
        os.makedirs(EVA_RESULT_DIR)

    # 初始化时间记录器
    ai_result_dirs = os.listdir(AI_RESULT_ROOT_DIR)
    case_pic_total = len(ai_result_dirs)
    t_log = TimeLogger(log, case_pic_total)

    # 记录所有图片的AI准确率评估值
    ai_correct_ratio_for_all = 0.0  # pylint: disable=invalid-name

    # dataframe for record error
    columns = [
        "标本编号",
        "图号",
        "错误说明",
        "轮廓差异度",
        "特征点相似度",
        "错误个数",
        "错误类型",
    ]
    err_df = pd.DataFrame(columns=columns)

    # 根据AI的识别结果目录，同GROUND TRUTH中的结果进行比对，计算准确率
    # 首先遍历AI识别结果目录，然后根据文件名找到对应的GROUND TRUTH目录，
    # 然后对GROUND TRUTH目录进行解析获得染色体分割和识别的标准答案，
    # 最后对比AI识别结果和染色体标准答案，进行匹配，计算准确率
    for ai_result_dir in ai_result_dirs:
        # 获取AI结果的案例号和图号目录的full path
        ai_result_dir_fp = os.path.join(AI_RESULT_ROOT_DIR, ai_result_dir)

        # 应该都是案例和图号的目录，文件就跳过
        if os.path.isfile(ai_result_dir_fp):
            continue

        log.info("  ")
        log.info(f"    vvvvvvvvvv  开始处理新CASE: {ai_result_dir}  vvvvvvvvvv")
        log.info("  ")

        case_id = ai_result_dir.split(".")[0]
        img_id = ai_result_dir.split(".")[1]

        t_log.case_started(ai_result_dir)

        ######################################
        # 开始解析AI识别结果中的染色体信息 #
        ######################################

        # 用于保存AI识别的染色体信息
        ai_chromo_result = []
        # 将AI结果按染色体序号chromo_idx进行组织
        ChromoData = Dict[str, Any]
        ai_chromo_orgby_chromo_idx: Dict[int, List[ChromoData]] = {}

        ai_chromo_result, ai_chromo_orgby_chromo_idx = get_chromo_info_from_result_dir(ai_result_dir_fp)

        log.info(f"AI推理结果获取完毕。AI分割识别出的染色体共 {len(ai_chromo_result)} 条。")

        ###############################################
        # 开始解析 GROUND TRUTH (gt) 中的染色体信息 #
        ###############################################

        log.info("开始解析 GROUND TRUTH 中的染色体信息")

        kyt_chromo_result = []
        kyt_chromo_orgby_chromo_idx: Dict[int, List[ChromoData]] = {}

        gt_result_dir_fp = os.path.join(GROUND_TRUTH_ROOT_DIR, ai_result_dir)

        if not os.path.exists(gt_result_dir_fp):
            # 如果在GROUND TRUTH中找不到对应的案例号和图号目录，就跳过
            continue

        kyt_chromo_result, kyt_chromo_orgby_chromo_idx = get_chromo_info_from_result_dir(gt_result_dir_fp)

        log.info(f"GROUND TRUTH解析完毕。GROUND TRUTH中的染色体共 {len(kyt_chromo_result)} 条。")

        ####################################################################
        # 开始对比核型报告图中的，和AI识别结果中的染色体信息，进行匹配 #
        ####################################################################

        # AI识别结果中染色体信息的数据结构 (ai_chromo_result, ai_chromo_orgby_chromo_idx)
        ## 本程序提供的信息
        ### [chromo_img, idx, id, cntr, roi, pos, bbox_bbg, bbox_wbg]
        # 报告图解析结果中染色体信息的数据结构 (kyt_chromo_result, kyt_chromo_orgby_chromo_idx)
        ## 报告图解析程序karyotype.py中的read_karyotype()函数返回的数据结构中chromosome的信息
        ### [cntr_idx, cntr, area, rect, bc_x, bc_y, bc_point, min_area_rect, cx, cy, center, chromo_idx(int), chromo_id, distance_to_id]
        ## 本程序补充的信息
        ### [bbox_bbg, bbox_wbg, pos]

        ai_correct_cnt_per_kyt = 0  # pylint: disable=invalid-name

        chromo_match_result_per_kyt: Dict[str, Any] = {}

        # 逐个编号进行匹配
        for chromo_idx in range(24):
            cur_ai_chromos = ai_chromo_orgby_chromo_idx[chromo_idx] if chromo_idx in ai_chromo_orgby_chromo_idx else []
            cur_kyt_chromos = (
                kyt_chromo_orgby_chromo_idx[chromo_idx] if chromo_idx in kyt_chromo_orgby_chromo_idx else []
            )
            cur_chromo_id = (  # pylint: disable=invalid-name
                str(chromo_idx + 1) if chromo_idx < 22 else ("X" if chromo_idx == 22 else "Y")
            )

            # 以kyt为基准，来检查AI的识别结果是否正确
            # 如果当前编号下kyt染色体和AI识别的染色体都有
            # 那么就用当前编号下kyt染色体去逐一匹配所有AI染色体找出最佳匹配
            # 如果找到最佳匹配的AI染色体也是当前编号下的，那么认为AI推理结果正确
            # 否则认为AI推理结果错误

            # 在匹配时，优先使用matchShapes方法，因为该方法很快，
            # 如果matchShapes方法匹配的最佳匹配染色体同kyt染色体编号一致，就认为AI推理结果正确；
            #     然后再用SIFT方法检测是否颠倒
            # 如果matchShapes方法匹配的最佳匹配染色体同kyt染色体编号不一致，那么就使用SIFT方法匹配
            # 如果SIFT方法匹配的最佳匹配染色体同kyt染色体编号一致，就认为AI推理结果正确；
            #    然后再用SIFT方法检测是否颠倒
            # 如果SIFT方法匹配的最佳匹配染色体同kyt染色体编号不一致(不再使用CLAHE增强后的SIFT方法匹配)，那么就认为AI推理结果错误

            if len(cur_kyt_chromos) > 0:

                # 逐一处理当前编号下的每条kyt染色体
                for cur_kyt_chromo in cur_kyt_chromos:

                    # 先用SIFT方法匹配
                    try:

                        (
                            sift_sim_score,
                            best_sift_match_ai_chromo,
                            _,
                            upside_down,
                        ) = best_feature_match_for_chromos(cur_kyt_chromo, ai_chromo_result)
                        # 如果SIFT方法匹配的最佳匹配染色体同kyt染色体编号一致，就认为AI推理结果正确
                        if best_sift_match_ai_chromo is not None and best_sift_match_ai_chromo["id"] == cur_chromo_id:

                            # 通过SIFT特征点BFMatcher匹配的AI染色体同kyt染色体的编号一致，认为AI推理结果正确
                            # 下面需要判断AI推理的染色体是否颠倒
                            if upside_down:
                                log.info(
                                    f"AI{best_sift_match_ai_chromo['id']}号染色体颠倒。位置: {best_sift_match_ai_chromo['pos']}。"
                                )
                                new_err = {
                                    "标本编号": case_id,
                                    "图号": img_id,
                                    "错误说明": f"AI{best_sift_match_ai_chromo['id']}号染色体颠倒。位置: {best_sift_match_ai_chromo['pos']}。",
                                    "轮廓差异度": f"{shape_diff_score:.2f}%",
                                    "特征点相似度": f"{sift_sim_score:.2f}%",
                                    "错误个数": 1,
                                    "错误类型": "极性",
                                }
                                err_df = pd.concat(
                                    [err_df, pd.DataFrame([new_err])],
                                    ignore_index=True,
                                )
                            else:
                                # 对正确的匹配进行计数
                                ai_correct_cnt_per_kyt += 1

                            # SIFT方法匹配的最佳匹配染色体同kyt染色体编号一致，AI推理结果正确，
                            # 不再使用其他方法匹配，直接跳过处理下一条染色体
                            continue

                    except Exception as e:

                        log.error(
                            f"计算kyt{cur_kyt_chromo['id']}号染色体同所有AI染色体进行SIFT相似度匹配时发生异常: {e}"
                        )
                        # 默认在终端打印异常, 默认颜色为红色
                        traceback.print_exc()
                        # 接收错误信息
                        err = traceback.format_exc()
                        log.error(err)

                    # 后用matchShapes方法匹配AI染色体
                    shape_diff_score, best_shape_match_ai_chromo = best_shape_match_for_chromos(
                        cur_kyt_chromo, ai_chromo_result
                    )

                    # 最佳匹配的AI染色体的编号同kyt染色体的编号一致，认为AI推理结果正确
                    if best_shape_match_ai_chromo is not None and best_shape_match_ai_chromo["id"] == cur_chromo_id:

                        # 通过matchShapes方法匹配的结果，认为AI推理结果正确
                        # 下面需要判断AI推理的染色体是否颠倒
                        try:

                            sift_sim_score, _, upside_down = feature_match_on_roi_for_flips(
                                cur_kyt_chromo["bbox_bbg"],
                                best_shape_match_ai_chromo["bbox_bbg"],
                            )

                            # 判断染色体是否颠倒
                            if upside_down:
                                # 染色体颠倒
                                log.info(
                                    f"AI{best_shape_match_ai_chromo['id']}号染色体颠倒。位置: {best_shape_match_ai_chromo['pos']}。"
                                )
                                new_err = {
                                    "标本编号": case_id,
                                    "图号": img_id,
                                    "错误说明": f"AI{best_shape_match_ai_chromo['id']}号染色体颠倒。位置: {best_shape_match_ai_chromo['pos']}。",
                                    "轮廓差异度": f"{shape_diff_score:.2f}%",
                                    "特征点相似度": f"{sift_sim_score:.2f}%",
                                    "错误个数": 1,
                                    "错误类型": "极性",
                                }
                                err_df = pd.concat(
                                    [err_df, pd.DataFrame([new_err])],
                                    ignore_index=True,
                                )
                            else:
                                # 对正确的匹配进行计数
                                ai_correct_cnt_per_kyt += 1

                        except Exception as e:
                            log.error(
                                f"计算kyt{cur_kyt_chromo['id']}号染色体同AI{best_shape_match_ai_chromo['id']}号染色体的SIFT相似度时发生异常: {e}"
                            )
                            # 默认在终端打印异常, 默认颜色为红色
                            traceback.print_exc()
                            # 接收错误信息
                            err = traceback.format_exc()  # pylint: disable=invalid-name
                            print(err)
                            log.error(err)

                    else:
                        # 先用SIFT特征匹配，后用matchShapes方法匹配，都匹配不上，就认为AI推理结果错误
                        # 匹配的染色体以SIFT方法匹配的结果为准
                        # 有可能在上面的SIFT方法匹配中，发生异常根本就没有给best_sift_match_ai_chromo赋值，这里用try except保护起来
                        try:
                            log.info(
                                f"{cur_kyt_chromo['id']}号染色体被AI识别为{best_sift_match_ai_chromo['id']}号染色体。"
                            )
                            new_err = {
                                "标本编号": case_id,
                                "图号": img_id,
                                "错误说明": f"{cur_kyt_chromo['id']}号染色体被AI识别为{best_sift_match_ai_chromo['id']}号染色体。",
                                "轮廓差异度": f"{shape_diff_score:.2f}%",
                                "特征点相似度": f"{sift_sim_score:.2f}%",
                                "错误个数": 1,
                                "错误类型": "识别",
                            }
                            err_df = pd.concat(
                                [err_df, pd.DataFrame([new_err])],
                                ignore_index=True,
                            )
                        except Exception as e:
                            log.error(
                                f"{cur_kyt_chromo['id']}号染色体通过SIFT算法匹配AI染色体时发生异常: {e}。以shape匹配结果为准。"
                            )
                            traceback.print_exc()
                            # 接收错误信息
                            err = traceback.format_exc()  # pylint: disable=invalid-name
                            print(err)
                            log.error(err)

                            # SIFT匹配方法发生异常，以matchShapes方法匹配的结果为准
                            log.info(
                                f"{cur_kyt_chromo['id']}号染色体被AI识别为{best_shape_match_ai_chromo['id']}号染色体。"
                            )
                            new_err = {
                                "标本编号": case_id,
                                "图号": img_id,
                                "错误说明": f"{cur_kyt_chromo['id']}号染色体被AI识别为{best_shape_match_ai_chromo['id']}号染色体。",
                                "轮廓差异度": f"{shape_diff_score:.2f}%",
                                "特征点相似度": "N/A",
                                "错误个数": 1,
                                "错误类型": "识别",
                            }
                            err_df = pd.concat(
                                [err_df, pd.DataFrame([new_err])],
                                ignore_index=True,
                            )

        # 计算当前案例下该图的评估结果
        ai_correct_ratio_per_kyt = (
            ai_correct_cnt_per_kyt / len(kyt_chromo_result) * 100 if len(kyt_chromo_result) > 0 else 0
        )
        log.info(f"{ai_result_dir}处理完毕, AI推理结果正确率: {ai_correct_ratio_per_kyt:.2f}%。")

        # 记录所有案例下，AI推理结果正确的染色体条数
        ai_correct_ratio_for_all += ai_correct_ratio_per_kyt

        log.info(" ")
        log.info("    ^^^^^^^^^^  处理完毕  ^^^^^^^^^^")
        log.info(" ")

        # 记录当前case+pic的时间
        t_log.case_finished(ai_result_dir)

    # 所有case+pic处理完毕记录总处理时间
    t_log.all_finished()

    # 所有案例下的所有报告图跑完了，计算平均准确率
    ai_correct_ratio_avg = ai_correct_ratio_for_all / case_pic_total if case_pic_total > 0 else 0
    log.info(f"所有案例下的报告图评估完毕。AI推理的平均准确率为 {ai_correct_ratio_avg:.2f}%")

    # 保存错误结果
    err_df_fp = os.path.join(EVA_RESULT_DIR, f"{CurrentFileName}_err_result_{time.strftime('%Y%m%d%H%M%S')}.xlsx")
    err_df.to_excel(err_df_fp, index=False)
