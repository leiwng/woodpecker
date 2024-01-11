# coding: utf8

"""batch_chromosome_matcher_v2 要解决的问题
- 在大图缩放到960x960的过程中，临近的轮廓会被"压缩"成一个，这样原算法以染色体标号单个字符的大小作为识别标志的方法就有问题了，两个字符会变成一个连通的轮廓，大小就变化了，按面积通过聚类来辨识染色体编号就会出问题。为了更广谱适用，改进的办法，换做用轮廓中心的纵坐标值辨识染色体编号字符轮廓。确定了编号轮廓就能确定那个染色体轮廓是属于那一对染色体的。
- 数据保存的问题，实验阶段跑一次数据的代价是比较高的，很多数据是在跑完后才意识到需要，而再跑一次的代价又很高，所以，需要在跑算法的过程中尽量把数据保存下来，以便后续使用。
- 为了把核型图/报告图中随体轮廓和染色体主体轮廓合并为一个轮廓，需要对染色体轮廓做垂直方向的膨胀，但不可避免地会有一定程度地水平膨胀，这样会造成两根靠得特别近地染色体连通变成一个轮廓。
- 为了广谱适用，不断探索调整尽量少的校准参数，以便达到最佳的效果。
- 对于靠近染色体的粘连问题可以用较低的BINARY_THRESHOLD值来把粘连处断开，然后再对染色体轮廓进行膨胀。
"""

import itertools
import random
import os
import math
import time

import traceback
from collections import namedtuple, Counter
from functools import partial
from turtle import shape
from cv2 import imwrite
from numba import jit
from operator import itemgetter

import cv2
import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


### 找外轮廓
def find_external_contours(img, bin_thresh=None):
    # 灰化
    dst_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if bin_thresh:
        # 二值化
        ret, dst_img = cv2.threshold(dst_img, bin_thresh, 255, cv2.THRESH_BINARY_INV)
    # 找轮廓
    contours, _ = cv2.findContours(dst_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours
    ### END of find_external_contours


### 根据轮廓数据，获取轮廓最小外接矩形，最小正矩形，最小外接矩形的面积，用于聚类的最小外接矩形的面积数组
def get_contour_info_list(contours):

    cnt_info = [{} for _ in range(len(contours))]
    for idx, cnt in enumerate(contours):

        cnt_info[idx]['cnt'] = cnt

        # ((cx, cy), (width, height), theta) = cv2.minAreaRect(cnt)
        minAR = cv2.minAreaRect(cnt)
        cnt_info[idx]['minAR'] = minAR

        rectCnt = np.int64(cv2.boxPoints(minAR))
        cnt_info[idx]['rectCnt'] = rectCnt

        # (x, y, w, h) = cv2.boundingRect(cnt)
        boundingRect = np.int64(cv2.boundingRect(cnt))
        cnt_info[idx]['boundingRect'] = boundingRect

        area = cv2.contourArea(rectCnt)
        cnt_info[idx]['area'] = area
        cnt_info[idx]['kArea'] = [7, area]

    return cnt_info
    ### END of get_contour_info_list


### 计算适合的KMean聚类的K值
def cal_k_of_KMeans(items):

    Scores = []
    max_k = 0
    max_s = 0
    for k in range(2, 9):
        estimator = KMeans(n_clusters=k)
        estimator.fit(items)
        score = silhouette_score(items, estimator.labels_, metric='euclidean')
        Scores.append(score)
        if score > max_s:
            max_s = score
            max_k = k

    return max_k
    ### END of cal_k_of_KMeans


### 计算两点间距离
def distance(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]

    return math.sqrt((x**2)+(y**2))
    ### END of distance


### 重新获取垂直膨胀后的轮廓
def contour_vertical_expansion(img_shape, contours, bin_img_thresh=245):

    if len(contours) == 0:
        raise(ValueError('FUNC: contour_vertical_expansion, param: contours is empty'))

    dst_img = np.zeros(img_shape, np.uint8)
    # 轮廓刻蚀
    for cnt in contours:
        cv2.drawContours(dst_img, [cnt], 0, (255,255,255), -1)

    # 垂直膨胀
    V_LINE = np.array([
        [0,1,0],
        [0,1,0],
        [0,1,0]], np.uint8)
    dst_img = cv2.dilate(dst_img, V_LINE, iterations=2)
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(dst_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 返回膨胀后的轮廓信息
    return contours
    ### END of contour_vertical_expansion


### 把轮廓中的内容分离出来,同时提供轮廓的mask图
def get_img_and_mask_from_contour(contour, src_img, bg_color=255):
    '''
    把轮廓从原图上把轮廓抠出来返回img,
    contour: 轮廓数据
    src_img: 原图，从原图上把轮廓抠出来
    bg_color: 背景色,缺省255
    '''
    img_clip = np.full_like(src_img, bg_color)
    img_mask = np.zeros_like(src_img)
    img_mask2 = np.zeros_like(src_img)

    # 在生成的img_mask上绘制contours的轮廓
    cv2.drawContours(img_mask, [contour], 0, 1, -1)
    cv2.drawContours(img_mask2, [contour], 0, (255,255,255), -1)

    # 得到包络轮廓的正矩形
    (x, y, w, h) = cv2.boundingRect(contour)

    # 割出mask图
    img_clip_mask = img_mask2[y:y+h, x:x+w]

    # mask out clip
    img_clip[img_mask == 1] = src_img[img_mask == 1]

    # 把染色体割出来
    img_clip = img_clip[y:y+h, x:x+w]
    # brd = SINGLE_CHROMO_BOX_BORDER
    # img_clip = cv2.copyMakeBorder(img_clip, brd, brd, brd, brd, cv2.BORDER_CONSTANT, value=0)

    return img_clip, img_clip_mask
    ### END of get_img_and_mask_from_contour


class RepImg:

    def __init__(self, img_fp):

        if img_fp is None:
            raise ValueError("Report Img init: img_fullpath is None")

        if not os.path.exists(img_fp):
            raise ValueError(f'Report Img init: image fullpath: {img_fp} does not exist')

        self.fp = img_fp
        (self.fpath, self.fname) = os.path.split(img_fp)
        [self.case_id, self.pic_id, self.f_type, self.f_ext] = self.fname.split('.')
        self.img = cv2.imread(self.fp)
        ### END OF __init__


    ### 判断轮廓面积是否符合染色体编号字符轮廓最小矩形的面积
    def is_chromo_id_area(self, area):

        return any(area > benchmark - 5 and area < benchmark + 5 for benchmark in self. ID_MINAR_AREA)
        ### END OF is_chromo_id_area


    ### 判断轮廓minAR角度是否符合染色体编号字符轮廓最小矩形的角度
    def is_chromo_id_angle(self, angle):

        return any(angle > benchmark - 1 and angle < benchmark + 1 for benchmark in self. ID_MINAR_ANGLE)
        ### END OF is_chromo_id_angle


    ### 判断轮廓是否是染色体编号上的横线
    def is_the_line_on_id_label(self, cnt_info):

        (w,h) = cnt_info['minAR'][1]
        hw_ratio = h//w if h > w else w//h

        return any(hw_ratio > limit - 5 and hw_ratio < limit + 5 for limit in self.ID_LINE_HW_RATIO)
        ### END OF is_the_line_on_id_label


    ### 保存报告图中的有效轮廓（去掉太小轮廓和无用的格式类轮廓）
    def __save_contour_info_list(self, contours):

        cnt_info_list = get_contour_info_list(contours)

        # 去掉过小的轮廓
        cnt_info_list = [ cnt for cnt in cnt_info_list if cnt['area'] > self.SMALL_CONTOUR_AREA ]

        # 去掉染色体编号上的横线
        cnt_info_list = [ cnt for cnt in cnt_info_list if not self.is_the_line_on_id_label(cnt) ]

        self.contour_info_list = cnt_info_list
        ### END of __save_contour_info_list


    ### 确定各行染色体编号的纵坐标 ID_ROW_CY
    def __calibrate_ID_ROW_CY(self):

        chromo_id_contours = [ cnt for cnt in self.contour_info_list if self.is_chromo_id_angle(cnt['minAR'][2]) and  self.is_chromo_id_area(cnt['area']) ]

        # 获取所有轮廓的纵坐标,判断有哪些纵坐标是染色体编号字符的纵坐标，从而基本确定报告图的行位置关系
        cnt_cy_list = [int(cnt['minAR'][0][1]) for cnt in chromo_id_contours]
        # 对相同的cy值计数
        cy_groupby_count = Counter(cnt_cy_list)

        rows_cy = []
        for cy, count in cy_groupby_count.items():

            if not rows_cy:
                rows_cy.append({'cy': cy, 'count': count})
                continue

            add_flag = True
            for item in rows_cy:
                if abs(item['cy'] - cy) < self.ID_ROW_CY_DIFF:
                    item['count'] += count
                    add_flag = False
                    break

            if add_flag:
                rows_cy.append({'cy': cy, 'count': count})

        rows_cy = sorted(rows_cy, key=lambda x: x['cy'], reverse=False)

        if len(rows_cy) != 4:
            # 报告图中的染色体不按照4行进行排列
            raise ValueError(f'Report Img: {self.fp} has {len(rows_cy)} rows, not 4 rows')

        for i in range(len(self.ID_CHAR_ROW_CNT)):
            if rows_cy[i]['count'] != self.ID_CHAR_ROW_CNT[i]:
                raise ValueError(f'Report Img: {self.fp} row:{i} has {rows_cy[i]["count"]} IDs, not {self.ID_CHAR_ROW_CNT[i]} IDs as benchmark')

        self.ID_ROW_CY = [item['cy'] for item in rows_cy]
        ### END OF __calibrate_ID_ROW_CY

    ### 对不同的报告图进行参数校准
    def img_property_calibration(self):

        ### 根据基准报告初始化 Image Properties Initialization

        # 二值化阈值
        self.BIN_IMG_THRESH = 245

        # 基准核型图中染色体编号单个字符最小矩形的中心坐标
        self.ID_ROW_CY = [242, 452, 614, 802]
        # 同一行染色体编号字符的纵坐标差值
        self.ID_ROW_CY_DIFF = 30

        # 基准核型图中染色体编号单个字符最小矩形的面积
        # 华西
        self.ID_1_MINAR_AREA = 104
        self.ID_2_9_MINAR_AREA = 117
        self.ID_X_Y_MINAR_AREA = 130
        self.ID_MINAR_AREA = [104, 117, 130]
        # 锦欣 2720 x 2048
        # self.ID_MINAR_AREA = [695, 714, 716, 726, 748, 759, 816, 825, 846, 858, 884]
        # 锦欣 960 x 960
        # self.ID_MINAR_AREA = [77,84,86,88,96,100,176,187,192,204]

        self.ID_MINAR_ANGLE = [0,26,27,28,29,30,31,90]

        # 基准核型图中染色体编号单个字符最小矩形的高宽
        self.ID_1_MINAR_HW = [13,8]
        self.ID_2_9_MINAR_HW = [13,9]
        self.ID_X_Y_MINAR_HW = [13,10]

        # 基准核型图中染色体编号单个字符的横坐标
        # 华西
        self.ID_CHAR_ROW_CX = [[] for _ in range(len(self.ID_ROW_CY))]
        self.ID_CHAR_ROW_CX[0] = [115, 330, 499, 759, 909]
        self.ID_CHAR_ROW_CX[1] = [57, 198, 340, 481, 619, 629, 760, 771, 901, 911]
        self.ID_CHAR_ROW_CX[2] = [53, 63, 194, 204, 336, 346, 618, 628, 759, 769, 901, 911]
        self.ID_CHAR_ROW_CX[3] = [53, 63, 193, 204, 405, 417, 546, 557, 764, 906]
        # 锦欣960x960
        # self.ID_CHAR_ROW_CX = [[] for _ in range(len(self.ID_ROW_CY))]
        # self.ID_CHAR_ROW_CX[0] = [115, 330, 499, 759, 909]
        # self.ID_CHAR_ROW_CX[1] = [57, 198, 340, 481, 619, 629, 760, 771, 911]
        # self.ID_CHAR_ROW_CX[2] = [53, 194, 336, 346, 618, 628, 759, 769, 911]
        # self.ID_CHAR_ROW_CX[3] = [53, 63, 204, 405, 417, 557, 764, 906]


        # 基准核型图中每排染色体编号的个数
        self.ID_ROW_CNT = [[] for _ in range(len(self.ID_ROW_CY))]
        self.ID_ROW_CNT[0] = 5
        self.ID_ROW_CNT[1] = 7
        self.ID_ROW_CNT[2] = 6
        self.ID_ROW_CNT[3] = 6

        # 需要去掉的太小轮廓上限
        self.SMALL_CONTOUR_AREA = 40

        # 基准核型图中两点的最大距离
        self.MAX_DISTANCE = 100000

        # 基准报告图背景色
        self.REP_BG_COLOR = [255, 255, 255]

        # 从基准报告图中割出来的单条染色体正矩形边框宽度
        self.SINGLE_CHROMO_BOX_BORDER = 3

        # 基准核型图中每排染色体编号单个字符的个数
        self.ID_CHAR_ROW_CNT = [[] for _ in range(len(self.ID_ROW_CY))]
        self.ID_CHAR_ROW_CNT[0] = len(self.ID_CHAR_ROW_CX[0]) # 5, 5
        self.ID_CHAR_ROW_CNT[1] = len(self.ID_CHAR_ROW_CX[1]) # 10, 9
        self.ID_CHAR_ROW_CNT[2] = len(self.ID_CHAR_ROW_CX[2]) # 12, 9
        self.ID_CHAR_ROW_CNT[3] = len(self.ID_CHAR_ROW_CX[3]) # 10, 8

        # 基准核型图中染色体编号单个字符轮廓的个数
        # 华西
        self.ID_CNT = 37
        # 锦欣 2720 x 2048
        # self.ID_CNT = 37
        # 锦欣 960 x 960
        # self.ID_CNT = sum(self.ID_CHAR_ROW_CNT)

        # 基准核型图中染色体编号上面横线宽高比
        self.ID_LINE_HW_RATIO = [119//3, 96//2, 96//1, 97//2]

        # 获取报告图的所有轮廓信息
        # 会去掉过小轮廓和染色体编号上的长横线轮廓
        self.__save_contour_info_list(find_external_contours(self.img, self.BIN_IMG_THRESH))

        # 确定各行染色体编号的纵坐标 ID_ROW_CY
        self.__calibrate_ID_ROW_CY()

        # 核型图的高宽
        self.IMG_H = self.img.shape[0]
        self.IMG_W = self.img.shape[1]

        # 基准核型图中染色体编号单个字符高宽
        # 华西
        self.ID_CHAR_H = 13
        self.ID_CHAR_W = 10
        # 锦欣 2720 x 2048
        # self.ID_CHAR_H = 34
        # self.ID_CHAR_W = 26
        # 锦欣 960 x 960
        # self.ID_CHAR_H = 12
        # self.ID_CHAR_W = 10

        # 字符高度的倍率
        self.ID_H_MF = 1
        self.MFed_ID_H = self.ID_CHAR_H * self.ID_H_MF

        # 核型图每行染色体和标号的行边界
        self.ID_ROW_ZONE_BORDER_Y = [[] for _ in range(len(self.ID_ROW_CY))]
        self.ID_ROW_ZONE_BORDER_Y[0] = {'top': 0,
                                        'floor': self.ID_ROW_CY[0] + self.MFed_ID_H}
        self.ID_ROW_ZONE_BORDER_Y[1] = {'top': self.ID_ROW_CY[0],
                                        'floor': self.ID_ROW_CY[1] + self.MFed_ID_H}
        self.ID_ROW_ZONE_BORDER_Y[2] = {'top': self.ID_ROW_CY[1],
                                        'floor': self.ID_ROW_CY[2] + self.MFed_ID_H}
        self.ID_ROW_ZONE_BORDER_Y[3] = {'top': self.ID_ROW_CY[2],
                                        'floor': self.ID_ROW_CY[3] + self.MFed_ID_H}

        self.ID_ROW_BORDER_CY = [[] for _ in range(len(self.ID_ROW_CY))]
        self.ID_ROW_BORDER_CY[0] = {'top':    self.ID_ROW_CY[0] - self.MFed_ID_H,
                                    'floor' : self.ID_ROW_CY[0] + self.MFed_ID_H}

        self.ID_ROW_BORDER_CY[1] = {'top':    self.ID_ROW_CY[1] - self.MFed_ID_H,
                                    'floor' : self.ID_ROW_CY[1] + self.MFed_ID_H}

        self.ID_ROW_BORDER_CY[2] = {'top':    self.ID_ROW_CY[2] - self.MFed_ID_H,
                                    'floor' : self.ID_ROW_CY[2] + self.MFed_ID_H}

        self.ID_ROW_BORDER_CY[3] = {'top':    self.ID_ROW_CY[3] - self.MFed_ID_H,
                                    'floor' : self.ID_ROW_CY[3] + self.MFed_ID_H}
        ### END OF img_property_calibration


    ### 对报告图中的轮廓最小矩形面积做聚类，得到各个轮廓聚类后的分类标注(labels)
    def __get_contour_size_category(self, cnt_info_list):

        cnt_areas = [cnt['kArea'] for cnt in cnt_info_list]
        k = cal_k_of_KMeans(cnt_areas)
        clusters = KMeans(n_clusters=k).fit(cnt_areas)

        return clusters.labels_
        ### END OF __get_contour_size_category


    ### 保存染色体编号轮廓信息
    def save_chromo_id_char_contour_info_list(self):

        labels = [ cnt['kLabel'] for cnt in self.contour_info_list ]
        # 各个聚类的个数
        count_groupby = Counter(labels)
        # 得到符合染色体编号字符轮廓数量的聚类
        # label:聚类编号，count:该聚类中的轮廓数量
        chromo_id_category_label_count = [ (label, count) for label, count in count_groupby.items() if count >= self.ID_CNT ]
        # 找到最符合染色体编号id轮廓的按个分类的label
        best_chromo_id_category_label = chromo_id_category_label_count[0][0]
        if len(chromo_id_category_label_count) > 1:
            max_chromo_id_cnt = 0
            for label_cnt in chromo_id_category_label_count:
                label = label_cnt[0]
                # 找聚类编号下对应轮廓的最小矩形面积接近染色体编号数字的面积
                chromo_id_cnt = 0
                for contour in self.contour_info_list:
                    if contour['kLabel'] == label:
                        area = contour['area']
                        if self.is_chromo_id_area(area):
                            chromo_id_cnt += 1
                        else:
                            continue
                if chromo_id_cnt > max_chromo_id_cnt:
                    max_chromo_id_cnt = chromo_id_cnt
                    best_chromo_id_category_label = label

        # 根据最符合的分类label找到染色体编号字符的轮廓信息
        self.chromo_id_char_contour_info_list = [ cnt for cnt in self.contour_info_list if cnt['kLabel'] == best_chromo_id_category_label ]

        # 保留符合染色体编号字符轮廓最小矩形面积的轮廓信息
        self.chromo_id_char_contour_info_list = [ cnt for cnt in self.chromo_id_char_contour_info_list if self.is_chromo_id_area(cnt['area']) ]

        # 保留符合染色体编号字符轮廓最小矩形角度的轮廓信息
        self.chromo_id_char_contour_info_list = [ cnt for cnt in self.chromo_id_char_contour_info_list if self.is_chromo_id_angle(cnt['minAR'][2]) ]
        ### END OF save_chromo_id_char_contour_info_list


    ### 保存报告图染色体编号信息
    def save_chromo_id_char_xy_info_rows(self):
        '''
        前置数据：
            报告图所有轮廓信息：self.cnt_info,
            cnt,minAR,rectCnt,boundingRect,area,kArea
        '''

        # 获取轮廓大小分类，通过分类来判断哪些轮廓是染色体编号轮廓
        contour_size_labels = self.__get_contour_size_category(self.contour_info_list)
        # 保存分类编号label

        if len(contour_size_labels) != len(self.contour_info_list):
            raise(ValueError(f'轮廓数量与分类数量不一致，轮廓数量：{len(self.contour_info_list)},分类数量：{len(contour_size_labels)}'))

        for idx in range(len(contour_size_labels)):
            self.contour_info_list[idx]['kLabel'] = contour_size_labels[idx]

        # 找到最符合染色体编号字符轮廓:self.chromo_id_char_contour_info_list
        self.save_chromo_id_char_contour_info_list()

        # 将染色体编号字符赶到每行中
        self.chromo_id_char_contour_info_rows = [ [] for _ in range(len(self.ID_ROW_CNT)) ]
        for i, cnt_info in itertools.product(range(len(self.chromo_id_char_contour_info_rows)), self.chromo_id_char_contour_info_list):
            cnt_cy = cnt_info['minAR'][0][1]
            top = self.ID_ROW_BORDER_CY[i]['top']
            floor = self.ID_ROW_BORDER_CY[i]['floor']
            if cnt_cy > top and cnt_cy < floor:
                self.chromo_id_char_contour_info_rows[i].append(cnt_info)

        # CHECK
        for i in range(len(self.chromo_id_char_contour_info_rows)):
            if len(self.chromo_id_char_contour_info_rows[i]) != len(self.ID_CHAR_ROW_CX[i]):
                raise(ValueError(f'染色体编号字符轮廓数量与染色体编号字符轮廓每行轮廓数量不一致，染色体编号字符轮廓数量：{len(self.chromo_id_char_contour_info_list)},染色体编号字符轮廓每行轮廓数量：{len(self.ID_CHAR_ROW_CX[i])}'))

        # 到这一步每行的染色体编号字符都已经按行存入row_chromo_id_char_cnt_info
        # 开始按行染色体编号字符提取信息
        self.chromo_id_char_xy_info_rows = [ [] for _ in range(len(self.ID_ROW_CNT)) ]
        for i in range(len(self.chromo_id_char_xy_info_rows)):
            for cnt_info in self.chromo_id_char_contour_info_rows[i]:
                a = {'cx': cnt_info['minAR'][0][0], 'cy': cnt_info['minAR'][0][1], 'cxy': cnt_info['minAR'][0]}
                self.chromo_id_char_xy_info_rows[i].append(a)

        # 染色体编号代表字符的坐标构成的数组，对于编号由两个字符构成的取左边的字符
        for i in range(len(self.chromo_id_char_xy_info_rows)):
            # 按x坐标排序
            id_char_xy_list = sorted(self.chromo_id_char_xy_info_rows[i], key=itemgetter('cx'))
            # 对于两个字符的染色体编号以左边的为准
            self.chromo_id_char_xy_info_rows[i] = []
            pre_left_id_char_cx = id_char_xy_list[0]['cx']
            cur_id_char_cx = id_char_xy_list[0]['cx']
            for idx in range(len(id_char_xy_list)):
                if idx == 0:
                    pre_left_id_char_cx = id_char_xy_list[idx]['cx']
                    self.chromo_id_char_xy_info_rows[i].append(id_char_xy_list[idx])
                    continue

                cur_id_char_cx = id_char_xy_list[idx]['cx']

                # 排除掉右侧的第二个字符
                if cur_id_char_cx - pre_left_id_char_cx < 2 * self.ID_CHAR_W:
                    pre_left_id_char_cx = cur_id_char_cx
                    continue

                # 保留正确的字符
                self.chromo_id_char_xy_info_rows[i].append(id_char_xy_list[idx])
                pre_left_id_char_cx = cur_id_char_cx

        # CHECK
        if len(self.chromo_id_char_xy_info_rows) != len(self.ID_ROW_CNT):
            raise(ValueError(f'实际染色体编号字符行数与基准报告图不一致，实际染色体编号行数：{len(self.chromo_id_char_xy_info_rows)},基准报告染色体编号行数：{len(self.ID_ROW_CNT)}'))

        for i in range(len(self.ID_ROW_CNT)):
            if len(self.chromo_id_char_xy_info_rows[i]) != self.ID_ROW_CNT[i]:
                raise(ValueError(f'当前行{i}，实际染色体编号代表字符数与基准报告图中染色体编号数量不一致，实际数量：{len(self.chromo_id_char_xy_info_rows[i])},基准报告图中的数量：{self.ID_ROW_CNT[i]}'))

        # 给每行的字符轮廓添加染色体编号
        chromo_num = 1
        for i, char_list in enumerate(self.chromo_id_char_xy_info_rows):
            for j, char in enumerate(char_list):
                # 先处理特殊的X，Y
                if i == len(self.chromo_id_char_xy_info_rows) - 1 and j == self.ID_ROW_CNT[i] - 2:
                    # X chromosome id, 最后一行倒数第二个字符
                    self.chromo_id_char_xy_info_rows[i][j]['chromo_id'] = 'X'
                    self.chromo_id_char_xy_info_rows[i][j]['chromo_num'] = 23
                    continue

                if i == len(self.chromo_id_char_xy_info_rows) - 1 and j == self.ID_ROW_CNT[i] - 1:
                    # Y chromosome id,最后一行倒数第一个字符
                    self.chromo_id_char_xy_info_rows[i][j]['chromo_id'] = 'Y'
                    self.chromo_id_char_xy_info_rows[i][j]['chromo_num'] = 24
                    continue

                # 处理其他染色体编号
                self.chromo_id_char_xy_info_rows[i][j]['chromo_id'] = str(chromo_num)
                self.chromo_id_char_xy_info_rows[i][j]['chromo_num'] = chromo_num

                chromo_num += 1
        ### END OF save_chromo_id_char_xy_info_rows


    ### 保存染色体轮廓信息
    def save_chromo_contour_info_list(self):

        # 从轮廓中去掉染色体编号字符的轮廓
        id_char_minARs = [ cnt['minAR'] for cnt in self.chromo_id_char_contour_info_list ]
        self.chromo_contour_info_list = [ cnt for cnt in self.contour_info_list if cnt['minAR'] not in id_char_minARs ]
        ### END OF save_chromo_contour_info_list


    ### 对染色体轮廓进行垂直膨胀，报告图中染色体头尾的碎片连成一体，并保存轮廓信息
    def save_dilated_chromo_contour_info_list(self):

        contours = [ cnt['cnt'] for cnt in self.chromo_contour_info_list ]
        # print(f'contours: {len(contours)}')

        self.dilated_chromo_contour_info_list = get_contour_info_list(contour_vertical_expansion(self.img.shape, contours, self.BIN_IMG_THRESH))
        ### END OF save_dilated_chromo_contour_info_list


    ### 将染色体同编号联系起来
    def link_chromo_with_id(self):

        # 染色体轮廓按行组织
        self.chromo_with_id_rows = [ [] for _ in range(len(self.ID_ROW_CY))]
        for i, border in enumerate(self.ID_ROW_ZONE_BORDER_Y):
            self.chromo_with_id_rows[i] = [ cnt for cnt in self.dilated_chromo_contour_info_list if cnt['minAR'][0][1] >= border['top'] and cnt['minAR'][0][1] <= border['floor'] ]

        # 把染色体编号同垂直膨胀后的染色体轮廓联系起来
        # self.chromo_id_char_xy_info_rows : 染色体编号字符info按行组织
        #
        # 按行循环,计算编号到染色体的距离，最小的就是该染色体所属的编号
        for i, (chromo_list, char_list) in enumerate(zip(self.chromo_with_id_rows, self.chromo_id_char_xy_info_rows)):

            # 对每行的每个染色体求该行各个编号的距离，该染色体属于距离最短的编号
            for chromo_idx, chromo in enumerate(chromo_list):

                chromo_p = chromo['minAR'][0]
                min_distance = self.MAX_DISTANCE
                the_chromo_id = '1'
                the_chromo_num = 1

                for char in char_list:
                    char_p = char['cxy']
                    the_distance = int(distance(chromo_p, char_p))
                    if the_distance < min_distance:
                        min_distance = the_distance
                        the_chromo_id = char['chromo_id']
                        the_chromo_num = char['chromo_num']

                self.chromo_with_id_rows[i][chromo_idx]['chromo_id'] = the_chromo_id
                self.chromo_with_id_rows[i][chromo_idx]['chromo_num'] = the_chromo_num

        # 每行按染色体编号排序
        for rox_idx in range(len(self.chromo_with_id_rows)):
            self.chromo_with_id_rows[rox_idx] = sorted(self.chromo_with_id_rows[rox_idx], key=itemgetter('chromo_num'))
        ### END of link_chromo_with_id


    ### 打印染色体和编号的对应关系
    def print_chromo_with_id_rows(self):
        for i in range(len(self.chromo_with_id_rows)):
            chromo_list = self.chromo_with_id_rows[i]
            for chromo in chromo_list:
                print(f"{i}:minAR:{chromo['minAR']},area:{chromo['area']},id:{chromo['chromo_id']},num:{chromo['chromo_num']}")
        ### END of print_chromo_with_id_rows


    ### 根据报告图的轮廓生成单根用于特征提取和匹配的染色体照片
    def save_single_chromo_img(self):
        # 将处理后的图片保存到self.chromo_with_id_rows中
        for i in range(len(self.chromo_with_id_rows)):
            for j in range(len(self.chromo_with_id_rows[i])):

                cnt = self.chromo_with_id_rows[i][j]['cnt']

                self.chromo_with_id_rows[i][j]['img'] = []
                self.chromo_with_id_rows[i][j]['img_mask'] = []

                img_clip, img_clip_mask = get_img_and_mask_from_contour(cnt, self.img)

                # 原图:0
                self.chromo_with_id_rows[i][j]['img'].append(img_clip)

                self.chromo_with_id_rows[i][j]['img_mask'].append(img_clip_mask)

                # 水平翻转:1
                self.chromo_with_id_rows[i][j]['img'].append(cv2.flip(img_clip, 1))

                self.chromo_with_id_rows[i][j]['img_mask'].append(cv2.flip(img_clip_mask, 1))

                # 垂直翻转:2
                self.chromo_with_id_rows[i][j]['img'].append(cv2.flip(img_clip, 0))

                self.chromo_with_id_rows[i][j]['img_mask'].append(cv2.flip(img_clip_mask, 0))

                # 水平垂直翻转:3
                self.chromo_with_id_rows[i][j]['img'].append(cv2.flip(img_clip, -1))

                self.chromo_with_id_rows[i][j]['img_mask'].append(cv2.flip(img_clip_mask, -1))

        ### END OF save_single_chromo_img
### END of RepImg Class


class OriImg:
    def __init__(self, img_fp):

        if img_fp is None:
            raise ValueError("Original Img init: Img Fullpath is None")

        if not os.path.exists(img_fp):
            raise ValueError(f'Original Img init: image fullpath: {img_fp} does not exist')

        self.fp = img_fp
        (self.fpath, self.fname) = os.path.split(img_fp)
        [self.case_id, self.pic_id, self.f_type, self.f_ext] = self.fname.split('.')
        self.img = cv2.imread(self.fp)

        # Image Properties Init
### END OF OriImg Class

### 特征匹配参数设置
# BINARY_THRESHOLD = 230
# CLOSING_KERNEL_SIZE = 6
# PAIR_CLOSING_KERNEL_SIZE = 20
# ROW_THRESHOLD = 15
# MERGE_CLOSING_KERNEL_SIZE = 15
# KEEPOUT_SIZE = 50
# DIST_THRESHOLD = 0.75
# MIN_GOOD_MATCHES = 3
# ROBUST_DIST_THRESHOLD = 0.9
AFFINE_GOOD_MATCHES = 3
ROBUST_GOOD_MATCHES = 4
CONTRAST_THRESHOLD = 0.01
EDGE_THRESHOLD = 20
SIGMA = 1.1
PAD = 1.2
### END of 特征匹配参数设置


def ransac_check(m, data, dist_threshold=1.0):
    if m is None:
        return False

    src, dst = data
    src_arr = np.float32([src]).reshape(-1, 1, 2)

    tgt = cv2.transform(src_arr, m)
    dist = np.linalg.norm(dst - tgt[0])

    return dist <= dist_threshold


def scale_check(m, data, scale_threshold=-0.1):
    if m is None:
        return False

    scale = math.sqrt(math.pow(m[0][0], 2) + math.pow(m[0][1], 2))

    return scale <= 1.2


def inline_check(m, data, dist_threshold=1.0, scale_threshold=0.1):
    return bool(ransac_check(m, data, dist_threshold=dist_threshold) and scale_check(m, data, scale_threshold=scale_threshold))


### ransac - Random Sample Consensus
def ransac(data, estimate_func, inliner_func, sample_size, max_iter, random_seed=None):
    best_inliner_count = 0
    best_inliners = []
    best_model = None

    random.seed(random_seed)

    for _ in range(max_iter):
        samples = random.sample(data, sample_size)
        model = estimate_func(samples)

        inliners = [inliner_func(model, d) for d in data]
        inliner_count = inliners.count(True)

        if inliner_count > best_inliner_count:
            best_inliner_count = inliner_count
            best_inliners = inliners
            best_model = model

    return best_model, best_inliners


def estimateAffineWrapper(samples):
    src_pts = np.float32([sample[0] for sample in samples]).reshape(-1, 1, 2)
    dst_pts = np.float32([sample[1] for sample in samples]).reshape(-1, 1, 2)

    m, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    return m


def estimateAffinePartial2D(src, dst, max_iter=500, ransac_threshold=1.0, scale_threshold=-0.1):
    inline_wrapper = partial(inline_check, dist_threshold=ransac_threshold, scale_threshold=scale_threshold)
    data = list(zip(src, dst))

    m, mask = ransac(data, estimateAffineWrapper, inline_wrapper, 3, max_iter=max_iter)

    return m, mask


class ChromoMatcher:

    ### 初始化

    global CONTRAST_THRESHOLD
    global EDGE_THRESHOLD
    global SIGMA
    global PAD

    def __init__(self, contrast_threshold=CONTRAST_THRESHOLD, edge_threshold=EDGE_THRESHOLD, sigma=SIGMA, pad=PAD, debug=False):

        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        self.pad = pad
        self.debug = debug
        # END of __init__


    ### 染色体同原图做特征匹配
    def _match(self, query, sample, dist_threshold=0.7, matcher='bf', feature='sift', match_box=None, draw=True):
        # query中是在原图中要匹配的染色体轮廓，sample中是原图
        # 通过xfeatures2d获取关键点
        if feature == 'sift':
            det = cv2.xfeatures2d.SIFT_create(
                contrastThreshold=self.contrast_threshold,
                edgeThreshold=self.edge_threshold,
                sigma=self.sigma
            )
        elif feature == 'surt':
            minHessian = 400
            det = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)

        if matcher == 'bf':
            matcher = cv2.BFMatcher()
        elif matcher == 'flann':
            matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

        kp1, des1 = det.detectAndCompute(query, None)

        # 局部匹配
        if match_box is not None:
            mask = np.zeros(sample.shape[:2], dtype='uint8')
            cv2.rectangle(mask, (match_box[0], match_box[1]), (match_box[2], match_box[3]), 255, -1)

            masked_sample = sample.copy()
            masked_sample = cv2.bitwise_and(masked_sample, mask)

            kp2, des2 = det.detectAndCompute(masked_sample, None)
        else:
            kp2, des2 = det.detectAndCompute(sample, None)

        matches = matcher.knnMatch(des1, des2, k=2)

        good = []
        good_without_list = []
        for m, n in matches:
            if m.distance < dist_threshold * n.distance:
                good.append([m])
                good_without_list.append(m)

        result = cv2.drawMatchesKnn(
            query,
            kp1,
            sample,
            kp2,
            good,
            None,
            flags=cv2.DrawMatchesFlags_DEFAULT
        )

        if draw:
            fig = plt.figure(figsize=(16, 12))
            plt.imshow(result)
            plt.show()

        return {'good': good, 'good_without_list': good_without_list, 'kp1': kp1, 'kp2': kp2, 'des1': des1, 'des2': des2, 'draw': result}
        ### END of _match


    ### 求透视矩阵M
    def _get_perspective(self, good_without_list, kp1, kp2, ransac_threshold=5.0, robust=True):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_without_list]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_without_list]).reshape(-1, 1, 2)

        if robust:
            perspective_matrix, m_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        else:
            perspective_matrix, m_mask = cv2.findHomography(src_pts, dst_pts, 0)

        match_mask = m_mask.ravel().tolist()

        return perspective_matrix, match_mask
        ### END of _get_perspective


    ### 求仿射矩阵M
    def _get_affine(self, good_without_list, kp1, kp2, offset_x=0., offset_y=0., ransac_threshold=3.0):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_without_list]).reshape(-1, 1, 2)
        src_pts -= np.float32([offset_x, offset_y])

        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_without_list]).reshape(-1, 1, 2)

        perspective_matrix, m_mask = estimateAffinePartial2D(src_pts, dst_pts, ransac_threshold=ransac_threshold, scale_threshold=-0.6)

        return perspective_matrix, m_mask
        ### END of _get_affine


    ### 求当根染色体到原图的仿射矩阵M
    def __get_M(self, match, robust=True, affine=True):

        if affine:
            m, m_mask = self._get_affine(match['good_without_list'], match['kp1'], match['kp2'])
        else:
            m, m_mask = self._get_perspective(match['good_without_list'], match['kp1'], match['kp2'])

        return m, m_mask
        ### END of __get_M


    ### choose best match single chromo image
    def __choose_best_match(self, match_list, affine=True):

        global AFFINE_GOOD_MATCHES
        global ROBUST_GOOD_MATCHES

        min_match = AFFINE_GOOD_MATCHES if affine else ROBUST_GOOD_MATCHES

        good_len_list = [ len(match['good']) for match in match_list ]

        qualified_good_cnt = sum(good_len >= min_match for good_len in good_len_list)

        if qualified_good_cnt == 0:
            return -1, None, None

        max_cnt = 0
        max_index = 0
        max_m = None
        max_m_mask = None
        for i, good_len in enumerate(good_len_list):
            if good_len >= min_match:
                m, m_mask = self.__get_M(match_list[i], robust=True, affine=affine)
                if m_mask.count(1) > max_cnt:
                    max_cnt = m_mask.count(1)
                    max_index = i
                    max_m = m
                    max_m_mask = m_mask

        return max_index, max_m, max_m_mask
        ### END of __choose_best_match_img


    ###
    def _warp_single_mask(self, mask_img, m, clip_mask, affine=True):

        all_zero = np.zeros_like(mask_img)

        h, w = mask_img.shape[:2]
        if affine:
            cv2.warpAffine(clip_mask, m, (w, h), all_zero)
        else:
            cv2.warpPerspective(clip_mask, m, (w, h), all_zero)

        return mask_img + all_zero
        ### END of _warp_single_mask


    ### 匹配报告图做所有染色体，以各种"姿势"(原图、水平、垂直、水平垂直)
    def match(self, ori_img, chromos_in_rows, draw=False, affine=True):

        for i in range(len(chromos_in_rows)):
            for j in range(len(chromos_in_rows[i])):

                # 对染色体的各自姿势就行匹配
                ch_img_list = chromos_in_rows[i][j]['img']
                ch_img_match_list = [self._match(ch_img, ori_img.img, draw=draw) for ch_img in ch_img_list]

                chromos_in_rows[i][j]['match'] = ch_img_match_list

                # choose best match image
                best_match_idx, best_match_m, best_match_m_mask = self.__choose_best_match(ch_img_match_list, affine=affine)

                # save match information
                # if best_match_idx != -1:
                chromos_in_rows[i][j]['best_match_idx'] = best_match_idx
                chromos_in_rows[i][j]['best_match_m'] = best_match_m
                chromos_in_rows[i][j]['best_match_m_mask'] = best_match_m_mask

        return chromos_in_rows
        ### END of match

### END OF ChromoMatcher Class


### DEBUG, 输出单个染色体的掩码图像
def dbg_write_chromo_clip_img(root_dir, fname, chromo_in_rows):

    # G2008311998.124.A.PNG
    [case_id, pic_id, f_type, ext] = fname.split('.')
    sub_dir = f'{case_id}.{pic_id}'
    dst_dir = os.path.join(root_dir, sub_dir)

    for i in range(len(chromo_in_rows)):
        for j in range(len(chromo_in_rows[i])):

            chromo_id = chromo_in_rows[i][j]['chromo_id'].zfill(2)
            chromo_num = str(chromo_in_rows[i][j]['chromo_num']).zfill(2)
            id2 = str(j).zfill(2)

            for n in range(len(chromo_in_rows[i][j]['img'])):
                fname = f'{case_id}.{pic_id}.{chromo_id}.{chromo_num}.{id2}.A{n}.png'
                cv2.imwrite(os.path.join(dst_dir, fname), chromo_in_rows[i][j]['img'][n])

            for n in range(len(chromo_in_rows[i][j]['img_mask'])):
                fname = f'{case_id}.{pic_id}.{chromo_id}.{chromo_num}.{id2}.M{n}.png'
                cv2.imwrite(os.path.join(dst_dir, fname), chromo_in_rows[i][j]['img_mask'][n])
    ### END of dbg_write_chromo_clip_img


### DEBUG,输出特征匹配图
def dbg_write_chromo_feature_match_img(root_dir, fname, chromo_in_rows):

    # G2008311998.124.A.PNG
    [case_id, pic_id, f_type, ext] = fname.split('.')
    sub_dir = f'{case_id}.{pic_id}'
    dst_dir = os.path.join(root_dir, sub_dir)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for i in range(len(chromo_in_rows)):
        for j in range(len(chromo_in_rows[i])):

            chromo_id = chromo_in_rows[i][j]['chromo_id'].zfill(2)
            chromo_num = str(chromo_in_rows[i][j]['chromo_num']).zfill(2)
            id2 = str(j).zfill(2)

            for n in range(len(chromo_in_rows[i][j]['match'])):
                match = chromo_in_rows[i][j]['match'][n]
                fname = f'{case_id}.{pic_id}.{chromo_id}.{chromo_num}.{id2}.FM{n}.png'
                cv2.imwrite(os.path.join(dst_dir, fname), match['draw'])
    ### END of dbg_write_chromo_feature_match_img


### 单张报告图到原图的染色体匹配
def chromo_matcher_from_rep_to_ori(output_root_dir, ori_fp, rep_fp):

    # 载入报告图
    rep_img = RepImg(rep_fp)

    # 根据输入报告图做图片参数校准
    rep_img.img_property_calibration()

    # 保存染色体编号字符信息
    rep_img.save_chromo_id_char_xy_info_rows()

    # 保存染色体信息
    rep_img.save_chromo_contour_info_list()

    # 保存垂直膨胀后的染色体信息
    rep_img.save_dilated_chromo_contour_info_list()

    # 将染色体同其编号联系起来
    rep_img.link_chromo_with_id()

    # Debug 参看由报告图生成的染色体和对应的编号信息
    # rep_img.print_chromo_with_id_rows()

    # 将报告图中的染色体切割成单个的图像保存(包括水平、垂直、水平垂直泛舟后的图像)
    rep_img.save_single_chromo_img()
    # self.chromo_with_id_rows[i][j]['img'][0:4] 原图,水平翻,垂直翻,水平垂直翻

    # 载入原图
    ori_img = OriImg(ori_fp)

    # 染色体特征匹配
    matcher = ChromoMatcher()

    ## 将报告图中的每一条染色体同原图进行特征匹配，希冀在原图中找到对应的染色体
    chromos_matcher_in_rows = matcher.match(ori_img, rep_img.chromo_with_id_rows)

    # Debug 保存单个染色体图片
    dbg_write_chromo_clip_img(output_root_dir, ori_img.fname, chromos_matcher_in_rows)
    # Debug 保存特征匹配图
    dbg_write_chromo_feature_match_img(output_root_dir, ori_img.fname, chromos_matcher_in_rows)

    # 生成罩回原图的掩码图
    for i in range(len(chromos_matcher_in_rows)):
        for j in range(len(chromos_matcher_in_rows[i])):
            if chromos_matcher_in_rows[i][j]['best_match_idx'] != -1:

                mask_img = np.zeros_like(ori_img.img)
                M = chromos_matcher_in_rows[i][j]['best_match_m']
                best_match_idx = chromos_matcher_in_rows[i][j]['best_match_idx']
                img_clip_mask = chromos_matcher_in_rows[i][j]['img_mask'][best_match_idx]

                mask_img = matcher._warp_single_mask(mask_img, M, img_clip_mask, affine=True)

                (fpath, fname) = os.path.split(ori_img.fp)

                # G2008311998.124.A.PNG
                [case_id, pic_id, f_type, ext] = fname.split('.')

                sub_dir = f'{case_id}.{pic_id}'
                dst_dir = os.path.join(output_root_dir, sub_dir)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)

                chromo_id = chromos_matcher_in_rows[i][j]['chromo_id'].zfill(2)
                chromo_num = str(chromos_matcher_in_rows[i][j]['chromo_num']).zfill(2)
                id2 = str(j).zfill(2)
                fname = f'{case_id}.{pic_id}.{chromo_id}.{chromo_num}.{id2}.png'

                dst_fp = os.path.join(dst_dir, fname)

                # print(dst_fp)

                cv2,imwrite(dst_fp, mask_img)

    ### END of chromo_matcher_from_rep_to_ori


if __name__ == '__main__':

    # ori_dir = 'D:\\labTable\\jinxin\\330\\png960x960'
    # rep_dir = 'D:\\labTable\\jinxin\\330\\png960x960'
    # output_root_dir = 'D:\\labTable\\jinxin\\330\\png960x960_output'

    ori_dir = 'D:\\Prj\\chromo-tech-test\\chromosome-matcher\\test\\test_data_4_patent'
    rep_dir = 'D:\\Prj\\chromo-tech-test\\chromosome-matcher\\test\\test_data_4_patent'
    output_root_dir = 'D:\\Prj\\chromo-tech-test\\chromosome-matcher\\test\\test_data_4_patent_output'

    if not os.path.exists(ori_dir):
        print(f'{ori_dir} not exists')
        exit(1)

    if not os.path.exists(rep_dir):
        print(f'{rep_dir} not exists')
        exit(2)

    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    for fname in os.listdir(rep_dir):
        # 220101001.004.K.png
        [case_id, pic_id, f_type, ext] = fname.split('.')
        if f_type != 'K':
            continue

        rep_fp = os.path.join(rep_dir, fname)
        ori_fp = os.path.join(ori_dir, f'{case_id}.{pic_id}.A.png')
        # t = time.time()

        try:
            chromo_matcher_from_rep_to_ori(output_root_dir, ori_fp, rep_fp)
        except Exception as e:
            print(f'{fname} error: {e}')
            print(e.args)
            print('=' * 20)
            print(traceback.format_exc())
            continue


