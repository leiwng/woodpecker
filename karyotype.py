'''
Module for Chromosome Karyotype Chart.

Usage:
    - Import this module using `import mymodule`.
    - Use the functions provided by this module as needed.

Author: Lei Wang
Date: Dec 14, 2023
'''
__author__ = "王磊"
__copyright__ = "Copyright 2023 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"


import os
import cv2
from copy import deepcopy
from sklearn.cluster import KMeans
import numpy as np
from utils.chromo_cv_utils import find_external_contours
import configparser


# 创建配置对象并读取配置文件
cfg = configparser.ConfigParser()
cfg.read('./config/karyotype.ini')

# BinThreshold = 253 ;求轮廓时图像二值化使用的阈值
BIN_THRESH = cfg.getint('General', 'BinThreshold')
# IdCharYTolerance = 4;同排染色体编号高度容差,单位像素
ID_CHAR_Y_TOLERANCE = cfg.getint('General', 'IdCharYTolerance')
# MaxIdCharArea = 80 ;染色体编号字符轮廓最大面积，实际测量为76，为了容差，设置为80
MAX_ID_CHAR_AREA = cfg.getint('General', 'MaxIdCharArea')
# 每排染色体编号最少字符数, 实际就是第1排染色体编号字符数, Row1IdCharNum = 5 ;第1排染色体编号字符数
ROW_ID_CHAR_MIN_NUM = cfg.getint('General', 'Row1IdCharNum')
# IdCharXTolerance = 40;染色体编号字符x坐标容差,单位像素,实测值为11,但同排不同染色体编号离得非常开,为了容差，设置为40
ID_CHAR_X_TOLERANCE = cfg.getint('General', 'IdCharXTolerance')
# TotalIdNum = 24 ;总染色体编号数
TOTAL_ID_NUM = cfg.getint('General', 'TotalIdNum')
# Row1IdNum = 5 ;第1排染色体编号数 1,2,3,4,5
ROW_1_ID_NUM = cfg.getint('General', 'Row1IdNum')
# Row2IdNum = 7 ;第2排染色体编号数 6,7,8,9,10,11,12
ROW_2_ID_NUM = cfg.getint('General', 'Row2IdNum')
# Row3IdNum = 6 ;第3排染色体编号数 13,14,15,16,17,18
ROW_3_ID_NUM = cfg.getint('General', 'Row3IdNum')
# Row4IdNum = 6 ;第4排染色体编号数 19,20,21,22,X,Y
ROW_4_ID_NUM = cfg.getint('General', 'Row4IdNum')


class Karyotype:
    '''
    Chromosome Karyotype Chart Class

    Attributes:
        attribute1 (int): An integer attribute.
        attribute2 (str): A string attribute.

    Methods:
        method1(): This method does something.
        method2(): This method does something else.

    Usage:
        - Create an instance of MyClass using `obj = MyClass()`.
        - Access attributes and call methods using `obj.attribute1`, `obj.method1()`, etc.
    '''

    def __init__(self, karyotype_img_fp):
        if karyotype_img_fp is None:
            raise ValueError('karyotype_img_fp is None')

        if not os.path.exists(karyotype_img_fp):
            raise FileNotFoundError(f'{karyotype_img_fp} is not exists')

        self.img = {}

        self.img['fp'] = karyotype_img_fp
        (self.img['fpath'], self.img['fname']) = os.path.split(self.img['fp'])
        (self.case_id, self.pic_id,
            self.img['type'], self.img['fext']) = self.img['fname'].split('.')

        self.img['img'] = cv2.imread(self.img['fp'])
        if self.img['img'] is None:
            raise ValueError(f'{karyotype_img_fp} is not a valid image')

        self.img['height'], self.img['width'], self.img['channels'] = self.img['img'].shape

        # member properties init
        self.contours_list = [] # 核型图中所有轮廓信息

        self.id_contours_cy_dict = {} # 核型图中染色体编号轮廓信息，按照cy为key进行组织
        self.id_contours_list = [] # 核型图中染色体编号轮廓信息

        self.id_char_contours_cy_dict = {} # 核型图中染色体编号字符轮廓信息
        self.id_char_contours_list = [] # 核型图中染色体编号字符轮廓信息


    def _id_info(self):
        """ Summary:
                获取核型图上染色体编号信息，该信息用于后续确定染色体编号;
                位于染色体下方，与染色体距离最近的编号，就是该染色体的编号。
            Member Properties Dependence:
                self.contours_list, # 核型图中所有轮廓信息
            Results:
                self.id_char_contours_info, # id_char_info (list of list of dict): [
                    [{'contour_idx':89,'contour':[[[x,y]]],...}, ...], # 第一排染色体编号字符轮廓信息
                    [{'contour_idx':73,'contour':[[[x,y]]],...}, ...], # 第二排染色体编号字符轮廓信息
                    [{'contour_idx':55,'contour':[[[x,y]]],...}, ...], # 第三排染色体编号字符轮廓信息
                    [{'contour_idx':41,'contour':[[[x,y]]],...}, ...], # 第四排染色体编号字符轮廓信息
                ]
                self.id_contours_info, # id_info (list of list of dict): [
                    [{'chromo_id':'1','chromo_idx':0,'cp':[x,y]}, ...], # 第一排染色体编号信息, cp为编号中心点坐标
                    [{'chromo_id':'6','chromo_idx':5,'cp':[x,y]}, ...], # 第二排染色体编号信息
                    [{'chromo_id':'13','chromo_idx':12,'cp':[x,y]}, ...], # 第三排染色体编号信息
                    [{'chromo_id':'19','chromo_idx':18,'cp':[x,y]}, ...], # 第四排染色体编号信息
                ]
        """
        # 先根据染色体编号字符的面积大小，过滤掉染色体轮廓
        id_cnts_info = [ cnt for cnt in self.contours_list if cnt['area'] <= MAX_ID_CHAR_AREA ]

        # 找到同排的染色体编号字符
        # 按照轮廓中心点cy坐标重新组织轮廓信息
        cy_dict = {}
        for cnt in id_cnts_info:
            key = cnt['cy']
            if key not in cy_dict:
                cy_dict[key] = []
            cy_dict[key].append(cnt)

        # 将cy差距小于等于同排染色体编号高度容差的轮廓合并为一组
        cy_dict_merge_result = deepcopy(cy_dict)
        for key in cy_dict:
            for key2 in cy_dict:
                if key != key2 and abs(key - key2) <= ID_CHAR_Y_TOLERANCE:
                    cy_dict_merge_result[key] = cy_dict_merge_result[key] + cy_dict_merge_result[key2]
                    del cy_dict_merge_result[key2]

        # 去掉轮廓数小于最小每排染色体编号字符数:ROW_ID_CHAR_MIN_NUM
        cy_dict_merge_result = { key:cy_dict_merge_result[key] for key in cy_dict_merge_result if len(cy_dict_merge_result[key]) < ROW_ID_CHAR_MIN_NUM }

        # 判断key个数是否为4，不为4则报错
        if len(cy_dict_merge_result) != 4:
            raise ValueError(f'{self.img["fp"]}染色体编号排的数量为{len(cy_dict_merge_result)},应该为4')

        # 按照cy坐标从小到大排序
        cy_dict_merge_result = dict(sorted(cy_dict_merge_result.items(), key=lambda item:item[0]))

        # SAVE RESULT to CLASS INSTANCE MEMBER PROPERTY
        self.id_char_contours_cy_dict = cy_dict_merge_result
        self.id_char_contours_list = [ cy_dict_merge_result[key] for key in cy_dict_merge_result ]

        # 把每排染色体编号字符x坐标距离小于等于ID_CHAR_X_TOLERANCE的只保留一个
        for key in cy_dict_merge_result:
            # 每排按照cx坐标从小到大排序
            cy_dict_merge_result[key] = sorted(cy_dict_merge_result[key], key=lambda item:item['cx'])
            pre_cx = None
            merged = []
            # 保留第一个
            for cnt in cy_dict_merge_result[key]:
                if pre_cx is None:
                    merged.append(cnt)
                    pre_cx = cnt['cx']
                else:
                    # 横向差距大于ID_CHAR_X_TOLERANCE的染色体编号字符轮廓才保留，编号是两个字符的，只保留左边的编号字符
                    if abs(cnt['cx'] - pre_cx) > ID_CHAR_X_TOLERANCE:
                        merged.append(cnt)
                        pre_cx = cnt['cx']
            cy_dict_merge_result[key] = merged

        # 轮廓经过案列合并和按行合并后，检查每行染色体编号的轮廓数
        id_num_in_rows = [ROW_1_ID_NUM, ROW_2_ID_NUM, ROW_3_ID_NUM, ROW_4_ID_NUM]
        for idx, key in enumerate(cy_dict_merge_result):
            if len(cy_dict_merge_result[key]) != id_num_in_rows[idx]:
                raise ValueError(f'{self.img["fp"]}第{idx+1}排染色体编号数量为{len(cy_dict_merge_result[key])},应该为{id_num_in_rows[idx]}')

        # 汇总染色体编号信息
        chromo_id_list = ["1","2","3","4","5",
            "6","7","8","9","10","11","12",
            "13","14","15","16","17","18",
            "19","20","21","22","X","Y"]
        chromo_idx = 0
        for key in cy_dict_merge_result:
            for cnt in cy_dict_merge_result[key]:
                cnt['chromo_id'] = chromo_id_list[chromo_idx]
                cnt['chromo_idx'] = chromo_idx
                chromo_idx += 1

        # SAVE RESULT to CLASS INSTANCE MEMBER PROPERTY
        self.id_contours_cy_dict = cy_dict_merge_result
        self.id_contours_list = [ cy_dict_merge_result[key] for key in cy_dict_merge_result ]


    def read_karyotype(self):
        """从报告图中读取染色体数据
        """
        # get all external contours
        contours = find_external_contours(
            self.img['img'], BIN_THRESH)

        # get all contours info
        self.contours_list = [{} for _ in range(len(contours))]
        for idx, contour in enumerate(contours):
            self.contours_list[idx]['contour_idx'] = idx
            self.contours_list[idx]['contour'] = contour
            self.contours_list[idx]['area'] = cv2.contourArea(contour)
            self.contours_list[idx]['rect'] = cv2.boundingRect(contour)
            self.contours_list[idx]['min_area_rect'] = cv2.minAreaRect(contour)
            self.contours_list[idx]['moments'] = cv2.moments(contour)
            self.contours_list[idx]['cx'] = int(
                self.contours_list[idx]['moments']['m10'] / self.contours_list[idx]['moments']['m00'])
            self.contours_list[idx]['cy'] = int(
                self.contours_list[idx]['moments']['m01'] / self.contours_list[idx]['moments']['m00'])
            self.contours_list[idx]['center'] = (
                self.contours_list[idx]['contour_center_x'], self.contours_list[idx]['contour_center_y'])

        # get chromo id info on karyotype chart
        # After this call, self.id_contours_list AND self.id_char_contours_list is ready to use
        # AND self.id_contours_cy_dict AND self.id_char_contours_cy_dict is also ready to use
        self._id_info()

        # get the left contours information except id contours
        # get id contours contour index
        id_char_contours_contour_idx = []
        for id_char_contours in self.id_char_contours_list:
            for cnt in id_char_contours:
                id_char_contours_contour_idx.append(cnt['contour_idx'])
        # get left contours
        left_contours_info = [ cnt for cnt in self.contours_list if cnt['contour_idx'] not in id_char_contours_contour_idx ]

        # organize left contours info by cy
        left_contours_cy_dict = {}
        cy_keys = self.id_contours_cy_dict.keys()
        for cnt in left_contours_info:
            top_y_limit = 0
            bottom_y_limit = cy_keys[0]
            for cy in cy_keys:
                bottom_y_limit = cy
                if cnt['cy'] > top_y_limit and cnt['cy'] < bottom_y_limit:
                    if cy not in left_contours_cy_dict:
                        left_contours_cy_dict[bottom_y_limit] = []
                    left_contours_cy_dict[bottom_y_limit].append(cnt)
                    break
                top_y_limit = bottom_y_limit
        # sort left contours dict by cx
        for key in left_contours_cy_dict:
            left_contours_cy_dict[key] = sorted(left_contours_cy_dict[key], key=lambda item:item['cx'])

        # match left contours with chromo id
        # with same row key: cy, match left contours with id contours
        # the contour in left contours belong to the nearest id contour
        for cy_key in left_contours_cy_dict:
            for left_cnt in left_contours_cy_dict[cy_key]:
                min_distance = float('inf')
                chromo_id = None
                chromo_idx = None
                for id_cnt in self.id_contours_cy_dict[cy_key]:
                    left_cnt_center = np.array(left_cnt['center'], dtype=np.int32)
                    id_cnt_center = np.array(id_cnt['center'], dtype=np.int32)
                    distance = np.linalg.norm(left_cnt_center - id_cnt_center)
                    if distance < min_distance:
                        min_distance = distance
                        chromo_id = id_cnt['chromo_id']
                        chromo_idx = id_cnt['chromo_idx']
                left_cnt['chromo_id'] = chromo_id
                left_cnt['chromo_idx'] = chromo_idx
                left_cnt['distance_to_id'] = min_distance


