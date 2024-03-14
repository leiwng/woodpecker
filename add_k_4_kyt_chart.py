import os


KYT_IMG_DIR = r"E:\染色体测试数据\240236-cjl给的AI准确率评估程序第二版测试数据\KYT_IMG"

for fn in os.listdir(KYT_IMG_DIR):
    # 把文件名按.分割，取所有元素
    items = fn.split(".")
    total_items = len(items)
    new_fn = "".join(
        f"{item}." if idx < total_items - 1 else f"K.{item}"
        for idx, item in enumerate(items)
    )
    old_fp = os.path.join(KYT_IMG_DIR, fn)
    new_fp = os.path.join(KYT_IMG_DIR, new_fn)
    os.rename(old_fp, new_fp)
