
import os
import shutil
import json

import cv2
import numpy as np


if __name__ == '__main__':

    src_dir = 'D:\\labTable\\jinxin\\330\\png960x960'
    mask_root_dir = 'D:\\labTable\\jinxin\\330\\png960x960_output'

    dst_dir = 'D:\\labTable\\jinxin\\330\\demo\\mask_img'

    for fname in os.listdir(mask_root_dir):

        mask_dir = os.path.join(mask_root_dir, fname)

        for fname in os.listdir(mask_dir):

            if len(fname.split('.')) == 6:

                # chromo_mask_img filename pattern: 220101001.004.0X.23.08.png
                [case_id, pic_id, chromo_id, chromo_num, col_id, f_ext] = fname.split('.')
                case_id = f'G{case_id}'

                chromo_id_dir = os.path.join(dst_dir, chromo_num)
                if not os.path.exists(chromo_id_dir):
                    os.makedirs(chromo_id_dir)

                # ori_img filename pattern : G2204010001.001.png
                # mask文件的文件名要和源文件名一致，通过不同的目录来分别存储不同的染色体mask文件
                chromo_mask_fname = f'{case_id}.{pic_id}.png'

                shutil.copyfile(os.path.join(mask_dir, fname), os.path.join(chromo_id_dir, chromo_mask_fname))

            else:
                continue