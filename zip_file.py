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

import zipfile
import os
import glob


def file2zip(zip_file_name: str, file_names: list):
    """将多个文件夹中文件压缩存储为zip
    :param zip_file_name:   /root/Document/test.zip
    :param file_names:      ['/root/user/doc/test.txt', ...]
    :return:
    """
    # 读取写入方式 ZipFile requires mode 'r', 'w', 'x', or 'a'
    # 压缩方式  ZIP_STORED： 存储； ZIP_DEFLATED： 压缩存储
    with zipfile.ZipFile(
        zip_file_name, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for fn in file_names:
            _, name = os.path.split(fn)

            # zipfile 内置提供的将文件压缩存储在.zip文件中， arcname即zip文件中存入文件的名称
            # 给予的归档名为 arcname (默认情况下将与 filename 一致，但是不带驱动器盘符并会移除开头的路径分隔符)
            zf.write(fn, arcname=name)

            # 等价于以下两行代码
            # 切换目录， 直接将文件写入。不切换目录，则会在压缩文件中创建文件的整个路径
            # os.chdir(parent_path)
            # zf.write(name)


if __name__ == "__main__":
    src_dir_fp = r"E:\染色体测试数据\240408-评估AI准确性_A2401150005-10_L2303060021-44_1550张\ORI_IMG"
    processed_cases = []

    for fn in os.listdir(src_dir_fp):
        if not fn.endswith(".png"):
            continue

        if len(fn.split(".")) != 3:
            continue

        case_id = fn.split(".")[0]
        img_id = fn.split(".")[1]

        if case_id in processed_cases:
            continue

        wildcard = f"{case_id}.*"
        wildcard_fp = os.path.join(src_dir_fp, wildcard)
        need_zip_files = glob.glob(wildcard_fp, recursive=False)
        zip_fn = f"{case_id}.zip"
        file2zip(zip_fn, need_zip_files)

        processed_cases.append(case_id)
