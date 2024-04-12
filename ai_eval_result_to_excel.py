# -*- coding: utf-8 -*-
"""
Module for convert AI result to Excel file

Usage:
    - Import this module using `import mymodule`.
    - Use the functions provided by this module as needed.

Author: Lei Wang
Date: Feb 8, 2024
"""


__author__ = "王磊"
__copyright__ = "Copyright 2023 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"


import os
import json
from openpyxl import Workbook


EVA_RESULT = r"E:\染色体测试数据\240204-评估240202测试集AI推理结果\EVA_RESULT\evaluate_ai_result_20240208020208_带染色体位置信息_包括极性错误_带报告图序号排序_to-Excel.json"
EVA_RESULT_DIR, EVA_RESULT_FN = os.path.split(EVA_RESULT)
EVA_RESULT_FNAME, EVA_RESULT_EXT = os.path.splitext(EVA_RESULT_FN)
CHROMO_ID_FIRST_CHAR = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "X", "Y"]

if __name__ == "__main__":

    wb = Workbook()
    ws = wb.active
    ws.title = "AI_Result"
    ws.append(["图号", "报告图染色体", "AI染色体", "相似度", "颠倒", "总体准确率评估值"])

    # ws2 = wb.create_sheet(title='Summary')

    with open(EVA_RESULT, "r", encoding="utf-8") as f:
        data = json.load(f)
        for case_id_pack in data:
            eva_details = []
            case_pic_id = ""
            acc_ratio = 0
            for key, value in case_id_pack.items():
                if key == "case_pic_id":
                    case_pic_id = value
                    continue
                if key == "acc_ratio":
                    acc_ratio = value
                    continue
                if key[0] in CHROMO_ID_FIRST_CHAR and (value["kyt_chromo_id"] != value["ai_chromo_id"] or value["UP_SIDE_DOWN"]):
                    kyt_chromo = f"{value["kyt_chromo_id"]}-{value["kyt_chromo_cx"]}"
                    ai_chromo = f"{value["ai_chromo_id"]}-{value["ai_chromo_position_idx"]}"
                    sim = value["similarity"]
                    up_down = "是" if value["UP_SIDE_DOWN"] else "否"
                    eva_details.append([kyt_chromo, ai_chromo, sim, up_down])

            if not eva_details:
                # AI结果全部正确
                ws.append([case_pic_id, "", "", "", "", acc_ratio])
            else:
                for eva_detail in eva_details:
                    ws.append([case_pic_id, eva_detail[0], eva_detail[1], eva_detail[2], eva_detail[3], acc_ratio])

    # save to Excel file
    excel_fn = f"{EVA_RESULT_FNAME}.xlsx"
    excel_fp = os.path.join(EVA_RESULT_DIR, excel_fn)
    wb.save(excel_fp)
