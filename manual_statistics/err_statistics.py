# -*- coding: utf-8 -*-
import sys


fn = "./result_merged.txt"
kwd = "号"
err_cnt_dict={"1":0, "2":0, "3":0, "4":0, "5":0, "6":0, "7":0, "8":0, "9":0, "10":0, "11":0, "12":0, "13":0, "14":0, "15":0, "16":0, "17":0, "18":0, "19":0, "20":0, "21":0, "22":0, "X":0, "Y":0}
with open(fn, 'r', encoding='utf-8') as f_obj:
    for line in f_obj:
        if kwd in line:
            chromo_id, cnt = line.split(kwd)
            _, cnt = cnt.split(": ")
            cnt = int(cnt.split("次")[0])
            err_cnt_dict[chromo_id] = err_cnt_dict.get(chromo_id, 0) + cnt

if not err_cnt_dict:
    print('没有找到关键字')
    sys.exit(3)

err_cnt_list = list(err_cnt_dict.items())
err_cnt_list.sort(key=lambda x: x[1], reverse=True)

for item in err_cnt_list:
    print(f'{item[0]}号染色体错误: {item[1]}次')
