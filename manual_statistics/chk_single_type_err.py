# -*- coding: utf-8 -*-
import sys
import chardet


def guess_encoding(file_path):
    """
    猜测文件编码
    :param file_path: 文件路径
    :return: 文件编码
    """
    with open(file_path, "rb") as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result["encoding"]


fn = "./simplified_error_annotate_23-12-11.txt"
kwd = "割"
# kwd = "认错"
# kwd = "颠倒"
kwds = ['割', '认错', '颠倒']

encoding = guess_encoding(fn)
print(f"File: {fn} detected encoding is: {encoding}")


with open(fn, "r", encoding="utf-8") as f_obj:

    err_cnt_dict = {}

    for kwd in kwds:

        for line in f_obj:
            if kwd in line:
                chromo_id = line.split(kwd)[0]
                err_cnt_dict[chromo_id] = err_cnt_dict.get(chromo_id, 0) + 1

        if not err_cnt_dict:
            print("没有找到关键字")
            sys.exit(3)

        err_cnt_list = list(err_cnt_dict.items())
        err_cnt_list.sort(key=lambda x: x[1], reverse=True)
        # print(f'"{kwd}"关键字统计：')

        for item in err_cnt_list:
            print(f"{item[0]}号染色体{kwd}错误: {item[1]}次")
