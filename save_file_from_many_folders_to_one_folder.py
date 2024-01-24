# -*- coding: utf-8 -*-
import os
import sys
from shutil import copyfile


def replace_separators(path):
    """_summary_
    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    return path.replace(os.sep, "=").replace("/", "=").replace("\\", "=")


def get_all_file_fullpath(dir_fp, fp_list):
    """遍历目录将所有的bmp文件转换为png文件

    Args:
        dir_fp (_type_): _description_
    """
    for entry in os.listdir(dir_fp):
        full_path = os.path.join(dir_fp, entry)
        if os.path.isdir(full_path):
            # is dir
            print(f"Entering {full_path} ...")
            get_all_file_fullpath(full_path, fp_list)  # Recursive call
        else:
            # is file
            # copy file to new dir
            print(f"screening {full_path} ...")
            fp_list.append(full_path)


if __name__ == "__main__":
    # 检查命令行参数大于1
    if len(sys.argv) < 3:
        print(
            "Usage: save_file_from_many_folders_to_one_folder.py <src_dir_path> <dst_dir_path"
        )
        sys.exit(0)

    # 获取命令行参数
    src_dir_fp = sys.argv[1]

    # 检查源目录路径字串是否是目录
    if not os.path.isdir(src_dir_fp):
        print(f"Error: {src_dir_fp} is not a directory!")
        sys.exit(0)

    # 检查源目录是否存在
    if not os.path.exists(src_dir_fp):
        print(f"Error: {src_dir_fp} does not exist!")
        sys.exit(0)

    dst_dir_fp = sys.argv[2]
    if not os.path.isdir(dst_dir_fp):
        print(f"Error: {dst_dir_fp} is not a directory!")
        sys.exit(0)

    if not os.path.exists(dst_dir_fp):
        # 不存在则创建
        os.makedirs(dst_dir_fp)

    src_fp_list = []
    get_all_file_fullpath(src_dir_fp, src_fp_list)

    common_src_fp_prefix = os.path.commonprefix(src_fp_list)
    print(f"common_src_fp_prefix: {common_src_fp_prefix}")

    for src_fp in src_fp_list:
        # 去掉源文件路径中的公共前缀
        dst_fp = src_fp[len(common_src_fp_prefix) :]
        dst_fp = replace_separators(dst_fp)
        dst_fp = os.path.join(dst_dir_fp, dst_fp)
        copyfile(src_fp, dst_fp)
