import os
from shutil import copyfile


ORI_IMG_DIR = (
    r"E:\染色体数据\240202_zzl_用于评估AI模型推理准确率的测试数据\测试数据\原图"
)

KYT_IMG_DIR = (
    r"E:\染色体数据\240202_zzl_用于评估AI模型推理准确率的测试数据\测试数据\核型图"
)

OUTPUT_ROOT_DIR = r"E:\染色体测试数据\240305-wl给zzl向绵阳妇幼演示的AI表现好的原图"

GOOD_LIST_FN = "original_img_of_good_AI_performance.txt"
GOOD_LIST_FP = os.path.join(OUTPUT_ROOT_DIR, GOOD_LIST_FN)

OUTPUT_ORI_IMG_DIR_FN = "GOOD_AI_PERFORMANCE_ORI_IMG"
OUTPUT_ORI_IMG_DIR_FP = os.path.join(OUTPUT_ROOT_DIR, OUTPUT_ORI_IMG_DIR_FN)

OUTPUT_KYT_IMG_DIR_FN = "GOOD_AI_PERFORMANCE_KYT_IMG"
OUTPUT_KYT_IMG_DIR_FP = os.path.join(OUTPUT_ROOT_DIR, OUTPUT_KYT_IMG_DIR_FN)


if not os.path.exists(OUTPUT_ORI_IMG_DIR_FP):
    os.makedirs(OUTPUT_ORI_IMG_DIR_FP)

if not os.path.exists(OUTPUT_KYT_IMG_DIR_FP):
    os.makedirs(OUTPUT_KYT_IMG_DIR_FP)

with open(GOOD_LIST_FP, "r", encoding="utf-8") as f:
    good_list = f.readlines()
    for line in good_list:
        if len(line) < 5:
            continue
        img_id = line.split("处理完毕")[0]
        ori_src_fn = f"{img_id}.png"
        ori_src_fp = os.path.join(ORI_IMG_DIR, ori_src_fn)
        ori_dst_fp = os.path.join(OUTPUT_ORI_IMG_DIR_FP, ori_src_fn)
        print(f"Copying {ori_src_fp} to {ori_dst_fp}")
        copyfile(ori_src_fp, ori_dst_fp)

        kyt_src_fn = f"{img_id}.K.JPG"
        kyt_src_fp = os.path.join(KYT_IMG_DIR, kyt_src_fn)
        kyt_dst_fp = os.path.join(OUTPUT_KYT_IMG_DIR_FP, kyt_src_fn)
        print(f"Copying {kyt_src_fp} to {kyt_dst_fp}")
        copyfile(kyt_src_fp, kyt_dst_fp)
