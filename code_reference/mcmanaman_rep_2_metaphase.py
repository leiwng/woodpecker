# 通过特征匹配,将报告图中割出来的染色体匹配到中期图中对应的染色体,然后
# 1. 保存中期图中的染色体图片和染色体编号信息
# 2. 保存染色体在中期图中的轮廓数据
# 3. 保存染色体在中期图中的掩码图
"""
- This version is based on the matcher version II. This version concise the algorithm of Feature Match and Affine Transformation from the chromosome in Karyotype chart to original photo image from camera
"""

import itertools
import json
import os
import sys
import traceback
from collections import Counter
from math import cos, fabs, radians, sin, sqrt
from operator import itemgetter

import cv2
import numpy as np
from matplotlib import pyplot as plt

from ai_eval_time_logger import TimeLogger

# global constant
# BIN_IMG_THRESH_4_CHROMO = 235
SMALL_CONTOUR_AREA_4_CHROMO = 10
WRAPPER_SIZE = 400

CHROMO_DIR = "chromo"
MASK_DIR = "mask"
DBG_DIR = "dbg"

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
DIST_THRESHOLD = 0.698
SIGMA = 1.1
PAD = 1.2
### END of 特征匹配参数设置


### get image background color
def get_bkg_color(img):
    """
    get_bkg_color: get the image background color
    """
    height, width = img.shape[:2]
    for h, w in itertools.product(range(height), range(width)):
        color = img[h, w]
        if (color == np.array([0, 0, 0])).all():
            continue
        else:
            return color.tolist()
    ### END of getImgBackgroundColor


# 根据height,width,channel number和颜色初始化画布
def init_canvas(width, height, channel_num, color=(255, 255, 255)):
    canvas = np.ones((height, width, channel_num), dtype="uint8")
    canvas[:] = color
    return canvas
    ### END of init_canvas


# 根据shape和颜色初始化画布
def init_canvas_from_shape(shape, color=(255, 255, 255)):
    canvas = np.ones(shape, dtype="uint8")
    canvas[:] = color
    return canvas
    ### END of init_canvas_from_shape


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

        cnt_info[idx]["cnt"] = cnt

        # ((cx, cy), (width, height), theta) = cv2.minAreaRect(cnt)
        minAR = cv2.minAreaRect(cnt)
        cnt_info[idx]["minAR"] = minAR

        rectCnt = np.int64(cv2.boxPoints(minAR))
        cnt_info[idx]["rectCnt"] = rectCnt

        # (x, y, w, h) = cv2.boundingRect(cnt)
        boundingRect = np.int64(cv2.boundingRect(cnt))
        cnt_info[idx]["boundingRect"] = boundingRect

        area = cv2.contourArea(cnt)
        cnt_info[idx]["area"] = area

        minARArea = cv2.contourArea(rectCnt)
        cnt_info[idx]["minARarea"] = minARArea

    return cnt_info
    ### END of get_contour_info_list


### 计算两点间距离
def distance(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]

    return sqrt((x**2) + (y**2))
    ### END of distance


### 计算两个矩形间的近似最短距离
def get_approx_min_dist_between_rect(rect1, rect2):
    min_dist = 999999
    nearest_pt1 = [0, 0]
    nearest_pt2 = [0, 0]
    for pt1 in rect1:
        for pt2 in rect2:
            dist = distance(pt1, pt2)
            if dist < min_dist:
                min_dist = dist
                nearest_pt1 = pt1
                nearest_pt2 = pt2
    return min_dist, nearest_pt1, nearest_pt2
    ### END of get_approx_min_dist_between_rect


### 计算两个轮廓间的近似最短距离
### 根据报告图上染色体和其随体的位置排列关系,用轮廓最小矩形4个角点坐标来计算轮廓间的距离
def get_approx_min_dist_between_contour(contour1, contour2):
    min_ar1 = cv2.minAreaRect(contour1)
    rect1 = np.int64(cv2.boxPoints(min_ar1))
    min_ar2 = cv2.minAreaRect(contour2)
    rect2 = np.int64(cv2.boxPoints(min_ar2))
    return get_approx_min_dist_between_rect(rect1, rect2)
    ### END of get_approx_min_dist_between_contour


### 计算两个轮廓间的最短距离,并返回距离最短的两个点的坐标[x, y],非常耗时,慎用!!!
def get_min_dist_between_contour(contour1, contour2):
    min_dist = 999999
    nearest_pt1 = [0, 0]
    nearest_pt2 = [0, 0]
    for pt1 in contour1:
        for pt2 in contour2:
            dist = distance(pt1[0], pt2[0])
            if dist < min_dist:
                min_dist = dist
                nearest_pt1 = pt1[0]
                nearest_pt2 = pt2[0]
    return min_dist, nearest_pt1, nearest_pt2
    ### END of get_min_dist_between_contour


### 以凸包的方式合并两个轮廓
def merge_contours(img_shape, contour1, contour2, nearest_pt1, nearest_pt2):

    img = np.zeros(img_shape, np.uint8)
    contours = [contour1, contour2]
    cv2.drawContours(img, contours, -1, (255, 255, 255), 1)

    cv2.line(img, nearest_pt1, nearest_pt2, (255, 255, 255), 1)

    new_contours = find_external_contours(img)
    return cv2.convexHull(new_contours[0])
    ### END of merge_contours


### 重新获取垂直膨胀后的轮廓
def contour_vertical_expansion(img_shape, contours, bin_img_thresh=245):

    if len(contours) == 0:
        raise (ValueError("FUNC: contour_vertical_expansion, param: contours is empty"))

    dst_img = np.zeros(img_shape, np.uint8)
    # 变成二值图,避免mask和原图叠加后又灰色的轮廓边缘
    # 不要用THRESH_BINARY_INV,会又反一遍,轮廓变黑,背景变白
    _, dst_img = cv2.threshold(dst_img, bin_img_thresh, 255, cv2.THRESH_BINARY)

    # 轮廓刻蚀
    cv2.drawContours(dst_img, contours, -1, (255, 255, 255), -1)

    # 垂直膨胀
    V_LINE = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8)
    dst_img = cv2.dilate(dst_img, V_LINE, iterations=2)

    # 返回膨胀后的轮廓信息
    return find_external_contours(dst_img)
    ### END of contour_vertical_expansion


### 把轮廓中的内容分离出来,同时提供轮廓的mask图
def get_chromo_img_and_mask_thru_contour_from_rep(contour, img, bin_thresh, bgc=255):
    """
    把轮廓从原图上把轮廓抠出来返回img,
    contour: 轮廓数据
    src_img: 原图，从原图上把轮廓抠出来
    bgc: 背景色,缺省255
    """

    # 导模
    mask = np.zeros(img.shape, np.uint8)
    # mask 二值化去除 包络染色体的线框
    # _, mask = cv2.threshold(mask, bin_thresh, 255, cv2.THRESH_BINARY)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
    # 取chromo
    chromo_clip = cv2.bitwise_and(img, mask)
    bBox = cv2.boundingRect(contour)
    (x, y, w, h) = bBox
    # 染色体
    chromo_clip = chromo_clip[y : y + h, x : x + w]
    # 染色体掩码
    chromo_clip_mask = mask[y : y + h, x : x + w]
    # 白背
    white_bk_chromo_clip = np.full_like(chromo_clip, bgc, dtype=np.uint8)
    np.copyto(white_bk_chromo_clip, chromo_clip, where=(chromo_clip_mask > 127))

    return white_bk_chromo_clip, chromo_clip_mask
    ### END of get_img_and_mask_from_contour


### 掩码图罩回原图抠出图像，并且符合以下要求：
### 摆正，白底，背景透明
def crop_img_from_mask(ori_img, mask_img, bin_thresh=240):

    # global BIN_IMG_THRESH_4_CHROMO
    global SMALL_CONTOUR_AREA_4_CHROMO, WRAPPER_SIZE

    # if not operator.eq(ori_img.shape[:2], mask_img.shape[:2]):
    #     raise(ValueError('FUNC: crop_img_from_mask, param: ori_img and mask_img shape not equal'))

    new_ori_img = ori_img.copy()
    new_mask_img = mask_img.copy()
    ori_img_bkg_color = get_bkg_color(ori_img)

    # 为了防止染色体贴边,抽正的时候被截断,先对mask和原图包边
    new_ori_img = cv2.copyMakeBorder(
        new_ori_img, WRAPPER_SIZE, WRAPPER_SIZE, WRAPPER_SIZE, WRAPPER_SIZE, cv2.BORDER_REPLICATE
    )
    new_mask_img = cv2.copyMakeBorder(
        new_mask_img, WRAPPER_SIZE, WRAPPER_SIZE, WRAPPER_SIZE, WRAPPER_SIZE, cv2.BORDER_REPLICATE
    )

    # 确保掩码图是二值的，否则染色体会有黑边
    _, new_mask_img = cv2.threshold(new_mask_img, bin_thresh, 255, cv2.THRESH_BINARY)

    # 通过掩码取图
    chromo_img = cv2.bitwise_and(new_ori_img, new_mask_img)

    # 白背
    white_bk_chromo_img = np.full_like(chromo_img, ori_img_bkg_color, dtype=np.uint8)
    np.copyto(white_bk_chromo_img, chromo_img, where=(new_mask_img > 127))

    # 取染色体轮廓，后面抽正用
    # 先把染色体按照正矩形割出来
    mask_contours = find_external_contours(new_mask_img)

    if len(mask_contours) == 0:
        raise (ValueError("FUNC: crop_img_from_mask, can not find contours in mask_img"))

    chromo_contour_in_ori = mask_contours[0]
    minAR = cv2.minAreaRect(chromo_contour_in_ori)
    # minBox = np.int64(cv2.boxPoints(minAR))

    cnt_center, cnt_size, cnt_angle = minAR[0], minAR[1], minAR[2]
    cnt_center, cnt_size = tuple(map(int, cnt_center)), tuple(map(int, cnt_size))
    (cnt_w, cnt_h) = cnt_size
    if cnt_w > cnt_h:
        cnt_angle = -(90 - cnt_angle)
        cnt_w, cnt_h = cnt_h, cnt_w

    # 取得旋转矩阵
    M = cv2.getRotationMatrix2D(cnt_center, cnt_angle, 1)

    img_h, img_w = white_bk_chromo_img.shape[:2]

    # 扩大画布,新的宽高，radians(angle) 把角度转为弧度 sin(弧度)
    new_h = int(img_w * fabs(sin(radians(cnt_angle))) + img_h * fabs(cos(radians(cnt_angle))))
    new_w = int(img_h * fabs(sin(radians(cnt_angle))) + img_w * fabs(cos(radians(cnt_angle))))

    # 旋转阵平移
    M[0, 2] += (new_w - img_w) / 2
    M[1, 2] += (new_h - img_h) / 2

    # 旋转
    rotated_chromo_img = cv2.warpAffine(white_bk_chromo_img, M, (new_w, new_h), borderValue=ori_img_bkg_color)
    rotated_mask_img = cv2.warpAffine(new_mask_img, M, (new_w, new_h), borderValue=[0, 0, 0])

    # 求旋转后的轮廓,这个求轮廓一定是用mask图,用chromo图在某些情况下会出两个轮廓
    # 主要原因是原图求轮廓会二值化,二值化后浅色区域会全白,造成染色体被割断
    new_contours = find_external_contours(rotated_mask_img)
    if len(new_contours) > 1:
        new_contours = [cnt for cnt in new_contours if cv2.contourArea(cnt) > SMALL_CONTOUR_AREA_4_CHROMO]

    # 把染色体抠出来形成小图
    new_contour = new_contours[0]
    box = cv2.boundingRect(new_contour)
    (x, y, w, h) = box
    rotated_chromo_img = rotated_chromo_img[y : y + h, x : x + w]

    # 背景透明
    rotated_chromo_img = cv2.cvtColor(rotated_chromo_img, cv2.COLOR_BGR2RGBA)
    white_pixels = np.where(
        (rotated_chromo_img[:, :, 0] == 255)
        & (rotated_chromo_img[:, :, 1] == 255)
        & (rotated_chromo_img[:, :, 2] == 255)
    )
    rotated_chromo_img[white_pixels] = [0, 0, 0, 0]

    # 图片被包边,需要把轮廓数据校正回未包边的情况
    for i in range(len(chromo_contour_in_ori)):
        chromo_contour_in_ori[i][0][0] -= WRAPPER_SIZE
        chromo_contour_in_ori[i][0][1] -= WRAPPER_SIZE

    return rotated_chromo_img, chromo_contour_in_ori


# 对摆正,白底,背景透明的染色体图像做左右开口旋转操作
def chromo_horizontal_flip(img, idx_in_pair):
    """
    :param img:传入的摆正,白底,背景透明的单根染色体图像
    :param idx_in_pair: 在染色体对中的索引, 0表示第一张, 1表示第二张
    :return: idx为0的开口向左,idx为其他值的开口向右
    """
    (h, w) = img.shape[:2]
    left_half = img[:, : w // 2]
    right_half = img[:, w // 2 :]

    if (idx_in_pair == 0 and left_half.mean() < right_half.mean()) or (
        idx_in_pair != 0 and left_half.mean() > right_half.mean()
    ):
        return cv2.flip(img, 1)
    return img


# 对摆正,白底,背景透明的染色体图像做垂直旋转操作
def chromo_stand_up(img):
    """
    :param img:传入的直立,白底,背景透明的单根染色体图像
    :return: 摆正后的染色体图片, 图片是否翻转过
    """
    (h, _, ch) = img.shape

    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if ch == 4 else img.copy()

    contours = find_external_contours(bgr_img, 240)

    mask_img = np.zeros(bgr_img.shape, np.uint8)
    cv2.drawContours(mask_img, contours, -1, (255, 255, 255), -1)

    top_half = mask_img[: h // 2, :]
    bottom_half = mask_img[h // 2 :, :]

    if top_half.mean() > bottom_half.mean():
        flipped = True
        return cv2.flip(img, 0), True

    return img, False


# 对摆正,白底,背景透明的染色体图像做垂直旋转操作
def chromo_vertical_flip(img):
    """
    :param img:传入的摆正,白底,背景透明的单根染色体图像
    :return: "头重脚轻"的单根染色体需要垂直翻转,因为是白色背景的图片所以上面不重就要颠倒
    """
    (h, w) = img.shape[:2]
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    top_half = bgr_img[: h // 2, :]
    bottom_half = bgr_img[h // 2 :, :]
    if top_half.mean() > bottom_half.mean():
        return cv2.flip(img, 0)
    return img


class RepImg:

    def __init__(self, img_fp):

        if img_fp is None:
            raise ValueError("Report Img init: img_fullpath is None")

        if not os.path.exists(img_fp):
            raise ValueError(f"Report Img init: image fullpath: {img_fp} does not exist")

        self.fp = img_fp
        (self.fpath, self.fname) = os.path.split(img_fp)
        [self.case_id, self.pic_id, self.f_type, self.f_ext] = self.fname.split(".")
        self.img = cv2.imread(self.fp)
        ### END OF __init__

    ### 判断轮廓是否是染色体编号上的横线
    def is_the_line_on_id_label(self, cnt_info):

        (w, h) = cnt_info["minAR"][1]
        hw_ratio = h // w if h > w else w // h

        return hw_ratio > self.ID_LINE_HW_RATIO_LL
        ### END OF is_the_line_on_id_label

    ### 判断轮廓中心y坐标是否在染色体编号字符宽度位置上
    def is_contour_on_id_label_bar_zone(self, cnt_info):
        cy = cnt_info["minAR"][0][1]
        return any(abs(cy - y) < self.MFed_ID_H for y in self.ID_ROW_CY)
        ### END OF is_contour_on_id_label_bar_zone

    ### 保存报告图中的有效轮廓（去掉太小轮廓和无用的格式类轮廓）
    def __save_contour_info_list(self, contours):

        cnt_info_list = get_contour_info_list(contours)

        # 去掉过小的轮廓
        cnt_info_list = [cnt for cnt in cnt_info_list if cnt["minARarea"] > self.SMALL_CONTOUR_AREA]

        # 去掉染色体编号上的横线
        cnt_info_list = [cnt for cnt in cnt_info_list if not self.is_the_line_on_id_label(cnt)]

        self.contour_info_list = cnt_info_list
        ### END of __save_contour_info_list

    ### 确定各行染色体编号的纵坐标 ID_ROW_CY
    def __calibrate_ID_ROW_CY(self):

        # 收集当前清理后所有轮廓最小矩形中心点的y坐标
        cys = [cnt["minAR"][0][1] for cnt in self.contour_info_list]
        cys_cnt_grpby = Counter(cys)

        # sort by key 既 (cy)
        sorted_cys_cnt_grpby = sorted(cys_cnt_grpby.items(), key=lambda x: x[0])

        # 由于每行染色体编号字符的中心点会有微小的偏差，
        # 所以需要合并容差以内的字典项
        merged_cys = {}
        pre_cy_k = 0
        for idx, (cy, cnt) in enumerate(sorted_cys_cnt_grpby):
            if idx == 0:
                merged_cys[cy] = cnt
                pre_cy_k = cy
                continue

            if abs(cy - pre_cy_k) < self.ID_ROW_CY_DIFF:
                merged_cys[pre_cy_k] += cnt
            else:
                merged_cys[cy] = cnt
                pre_cy_k = cy

        # 取相同坐标计数值大于等于5的项
        filtered_merged_cys = {k: v for k, v in merged_cys.items() if v >= self.ID_CHAR_ROW_CNT[0]}

        # 也有染色体的中心y值比较平直的情况，当两行y值的差比较接近的时候，合并到y值较大的那个
        while len(filtered_merged_cys) > 4:
            cnt = 0
            pre_cy_k = 0
            pre_cnt_v = 0
            distances = []
            for k, v in filtered_merged_cys.items():
                distance = int(k - pre_cy_k)
                distances.append(
                    {"dis": distance, "up_cy_k": pre_cy_k, "up_cnt_v": pre_cnt_v, "down_cy_k": k, "down_cnt_v": v}
                )
                pre_cy_k = k
                pre_cnt_v = v
            min_dis = min(distances, key=lambda x: x["dis"])
            filtered_merged_cys.pop(min_dis["up_cy_k"])

        if len(filtered_merged_cys) < 4:
            raise ValueError(
                f"{self.fp} ID_ROW_CY calibration failed, calculated number of chromosome id row is less 4."
            )

        filtered_merged_cys = list(map(int, filtered_merged_cys.keys()))
        self.ID_ROW_CY = filtered_merged_cys
        ### END of __calibrate_ID_ROW_CY

    ### 对不同的报告图进行参数校准
    def img_property_calibration(self):

        ### 根据基准报告初始化 Image Properties Initialization

        # 二值化阈值
        # 取230尽量避免，同一对染色体在报告图上的粘连，
        # 以及染色体图报告图上染色体编号上面横线的粘连，
        # 在生成单个染色体的掩码图后，再对掩码图进行膨胀，
        # 以尽量用掩码图抠取完整的染色体。
        self.IMG_BIN_THRESH = 240

        # 需要去掉的太小轮廓上限
        # 如果原图分辨率很高,比如2720x2048,即使是很小的轮廓他的最小矩形的面积也会很大
        # 对于锦欣960x960的数据，最小染色体碎片轮廓的最小矩形面积为在635
        # jinxin
        # self.SMALL_CONTOUR_AREA = 55
        # huaxi 有把小的随体放到染色体主体轮廓上的情况
        self.SMALL_CONTOUR_AREA = 20

        # 基准核型图中两点的最大距离
        self.MAX_DISTANCE = 100000

        # 基准核型图中染色体和随体(satellite)的最小距离上限
        self.MAX_DISTANCE_BETWEEN_CHROMO_AND_ITS_SATELLITE = 10

        # 基准核型图中染色体和随体(satellite)面积比率的阈值
        self.AREA_RATIO_BETWEEN_CHROMO_AND_SATE = 3

        # 基准核型图中染色体编号单个字符最小矩形的中心坐标
        self.ID_ROW_CY = [242, 452, 614, 802]
        # 同一行染色体编号字符的纵坐标差值
        self.ID_ROW_CY_DIFF = 1.5

        # 基准核型图中每排染色体编号的个数
        self.ID_ROW_CNT = [[] for _ in range(len(self.ID_ROW_CY))]
        self.ID_ROW_CNT[0] = 5
        self.ID_ROW_CNT[1] = 7
        self.ID_ROW_CNT[2] = 6
        self.ID_ROW_CNT[3] = 6

        # 基准核型图中每排染色体编号单个字符的个数
        self.ID_CHAR_ROW_CNT = [[] for _ in range(len(self.ID_ROW_CY))]
        self.ID_CHAR_ROW_CNT[0] = 5  # 5, 5
        # self.ID_CHAR_ROW_CNT[1] = 10 # 10, 9
        # self.ID_CHAR_ROW_CNT[2] = 12 # 12, 9
        # self.ID_CHAR_ROW_CNT[3] = 10 # 10, 8

        # 基准核型图中染色体编号上面横线宽高比
        # self.ID_LINE_HW_RATIO = [119//3, 96//2, 96//1, 97//2]
        self.ID_LINE_HW_RATIO_LL = 30

        # 获取报告图的所有轮廓信息
        # 会去掉过小轮廓和染色体编号上的长横线轮廓
        self.__save_contour_info_list(find_external_contours(self.img, self.IMG_BIN_THRESH))

        # 确定各行染色体编号的纵坐标 ID_ROW_CY
        self.__calibrate_ID_ROW_CY()

        # 核型图的高宽
        # self.IMG_H = self.img.shape[0]
        # self.IMG_W = self.img.shape[1]

        # 基准核型图中染色体编号单个字符高宽
        # 随不同医院和不同报告图分辨率不同而不同
        # jinxin 320band
        # self.ID_CHAR_H = 15
        # self.ID_CHAR_W = 14
        # huaxi 320band
        self.ID_CHAR_H = 14
        self.ID_CHAR_W = 10

        # 字符高度的倍率
        self.ID_H_MF = 1
        self.MFed_ID_H = self.ID_CHAR_H * self.ID_H_MF

        # 核型图每行染色体和标号的行边界
        self.ID_ROW_ZONE_BORDER_Y = [[] for _ in range(len(self.ID_ROW_CY))]
        self.ID_ROW_ZONE_BORDER_Y[0] = {"top": 0, "floor": self.ID_ROW_CY[0] + self.MFed_ID_H}
        self.ID_ROW_ZONE_BORDER_Y[1] = {"top": self.ID_ROW_CY[0], "floor": self.ID_ROW_CY[1] + self.MFed_ID_H}
        self.ID_ROW_ZONE_BORDER_Y[2] = {"top": self.ID_ROW_CY[1], "floor": self.ID_ROW_CY[2] + self.MFed_ID_H}
        self.ID_ROW_ZONE_BORDER_Y[3] = {"top": self.ID_ROW_CY[2], "floor": self.ID_ROW_CY[3] + self.MFed_ID_H}

        self.ID_ROW_BORDER_CY = [[] for _ in range(len(self.ID_ROW_CY))]
        self.ID_ROW_BORDER_CY[0] = {
            "top": self.ID_ROW_CY[0] - self.MFed_ID_H,
            "floor": self.ID_ROW_CY[0] + self.MFed_ID_H,
        }

        self.ID_ROW_BORDER_CY[1] = {
            "top": self.ID_ROW_CY[1] - self.MFed_ID_H,
            "floor": self.ID_ROW_CY[1] + self.MFed_ID_H,
        }

        self.ID_ROW_BORDER_CY[2] = {
            "top": self.ID_ROW_CY[2] - self.MFed_ID_H,
            "floor": self.ID_ROW_CY[2] + self.MFed_ID_H,
        }

        self.ID_ROW_BORDER_CY[3] = {
            "top": self.ID_ROW_CY[3] - self.MFed_ID_H,
            "floor": self.ID_ROW_CY[3] + self.MFed_ID_H,
        }
        ### END OF img_property_calibration

    ### 保存染色体编号轮廓信息
    def save_chromo_id_char_contour_info_list(self):

        # 为了普适性，判断是否是染色体编号字符轮廓就只有
        # 一条标准：根据轮廓中心的y坐标进行判断
        # 坐标的校准：__calibrate_ID_ROW_CY
        # self.ID_ROW_CY
        self.chromo_id_char_contour_info_list = [
            cnt_info for cnt_info in self.contour_info_list if self.is_contour_on_id_label_bar_zone(cnt_info)
        ]
        ### END OF save_chromo_id_char_contour_info_list

    ### 保存报告图染色体编号信息
    def save_chromo_id_char_xy_info_rows(self):
        """
        前置数据：
            报告图所有轮廓信息:self.cnt_info,
            cnt,minAR,rectCnt,boundingRect,area
        """

        # 找到最符合染色体编号字符轮廓:self.chromo_id_char_contour_info_list
        self.save_chromo_id_char_contour_info_list()

        # 将染色体编号字符赶到每行中
        self.chromo_id_char_contour_info_rows = [[] for _ in range(len(self.ID_ROW_CNT))]
        for i, cnt_info in itertools.product(
            range(len(self.chromo_id_char_contour_info_rows)), self.chromo_id_char_contour_info_list
        ):
            cnt_cy = cnt_info["minAR"][0][1]
            top = self.ID_ROW_BORDER_CY[i]["top"]
            floor = self.ID_ROW_BORDER_CY[i]["floor"]
            if cnt_cy > top and cnt_cy < floor:
                self.chromo_id_char_contour_info_rows[i].append(cnt_info)

        # 到这一步每行的染色体编号字符都已经按行存入row_chromo_id_char_cnt_info
        # 开始按行染色体编号字符提取信息
        self.chromo_id_char_xy_info_rows = [[] for _ in range(len(self.ID_ROW_CNT))]
        for i in range(len(self.chromo_id_char_xy_info_rows)):
            for cnt_info in self.chromo_id_char_contour_info_rows[i]:
                a = {"cx": cnt_info["minAR"][0][0], "cy": cnt_info["minAR"][0][1], "cxy": cnt_info["minAR"][0]}
                self.chromo_id_char_xy_info_rows[i].append(a)

        # 染色体编号代表字符的坐标构成的数组，对于编号由两个字符构成的取左边的字符
        for i in range(len(self.chromo_id_char_xy_info_rows)):
            # 按x坐标排序
            id_char_xy_list = sorted(self.chromo_id_char_xy_info_rows[i], key=itemgetter("cx"))
            # 对于两个字符的染色体编号以左边的为准
            self.chromo_id_char_xy_info_rows[i] = []
            pre_left_id_char_cx = id_char_xy_list[0]["cx"]
            cur_id_char_cx = id_char_xy_list[0]["cx"]
            for idx in range(len(id_char_xy_list)):
                if idx == 0:
                    pre_left_id_char_cx = id_char_xy_list[idx]["cx"]
                    self.chromo_id_char_xy_info_rows[i].append(id_char_xy_list[idx])
                    continue

                cur_id_char_cx = id_char_xy_list[idx]["cx"]

                # 排除掉右侧的第二个字符
                if cur_id_char_cx - pre_left_id_char_cx < 2 * self.ID_CHAR_W:
                    pre_left_id_char_cx = cur_id_char_cx
                    continue

                # 保留正确的字符
                self.chromo_id_char_xy_info_rows[i].append(id_char_xy_list[idx])
                pre_left_id_char_cx = cur_id_char_cx

        # CHECK
        if len(self.chromo_id_char_xy_info_rows) != len(self.ID_ROW_CNT):
            raise (
                ValueError(
                    f"实际染色体编号字符行数与基准报告图不一致，实际染色体编号行数：{len(self.chromo_id_char_xy_info_rows)},基准报告染色体编号行数：{len(self.ID_ROW_CNT)}"
                )
            )

        for i in range(len(self.ID_ROW_CNT)):
            if len(self.chromo_id_char_xy_info_rows[i]) != self.ID_ROW_CNT[i]:
                raise (
                    ValueError(
                        f"当前行{i}，实际染色体编号代表字符数与基准报告图中染色体编号数量不一致，实际数量：{len(self.chromo_id_char_xy_info_rows[i])},基准报告图中的数量：{self.ID_ROW_CNT[i]}"
                    )
                )

        # 给每行的字符轮廓添加染色体编号
        chromo_num = 1
        for i, char_list in enumerate(self.chromo_id_char_xy_info_rows):
            for j, char in enumerate(char_list):
                # 先处理特殊的X，Y
                if i == len(self.chromo_id_char_xy_info_rows) - 1 and j == self.ID_ROW_CNT[i] - 2:
                    # X chromosome id, 最后一行倒数第二个字符
                    self.chromo_id_char_xy_info_rows[i][j]["chromo_id"] = "X"
                    self.chromo_id_char_xy_info_rows[i][j]["chromo_num"] = 23
                    continue

                if i == len(self.chromo_id_char_xy_info_rows) - 1 and j == self.ID_ROW_CNT[i] - 1:
                    # Y chromosome id,最后一行倒数第一个字符
                    self.chromo_id_char_xy_info_rows[i][j]["chromo_id"] = "Y"
                    self.chromo_id_char_xy_info_rows[i][j]["chromo_num"] = 24
                    continue

                # 处理其他染色体编号
                self.chromo_id_char_xy_info_rows[i][j]["chromo_id"] = str(chromo_num)
                self.chromo_id_char_xy_info_rows[i][j]["chromo_num"] = chromo_num

                chromo_num += 1
        ### END OF save_chromo_id_char_xy_info_rows

    ### 保存染色体轮廓信息
    def save_chromo_contour_info_list(self):

        # 从轮廓中去掉染色体编号字符的轮廓
        id_char_minARs = [cnt["minAR"] for cnt in self.chromo_id_char_contour_info_list]
        self.chromo_contour_info_list = [cnt for cnt in self.contour_info_list if cnt["minAR"] not in id_char_minARs]
        ### END OF save_chromo_contour_info_list

    ### 对染色体轮廓进行垂直膨胀，报告图中染色体头尾的碎片连成一体，并保存轮廓信息
    def save_dilated_chromo_contour_info_list(self):

        contours = [cnt["cnt"] for cnt in self.chromo_contour_info_list]
        # print(f'contours: {len(contours)}')

        self.dilated_chromo_contour_info_list = get_contour_info_list(
            contour_vertical_expansion(self.img.shape, contours, self.IMG_BIN_THRESH)
        )
        ### END OF save_dilated_chromo_contour_info_list

    ### 将染色体和其随体(satellite)合并到同一轮廓
    def merge_chromo_and_its_satellite(self):

        chromo_and_its_satellite = []
        for cnt_info1_idx, cnt_info1 in enumerate(self.dilated_chromo_contour_info_list):
            min_cnt_dist = 99999
            min_dist_cnt_info1 = None
            min_dist_cnt_info2 = None
            min_dist_cnt_info1_idx = -1
            min_dist_cnt_info2_idx = -1
            min_dist_nearest_pt1 = None
            min_dist_nearest_pt2 = None
            found_close_contours = False
            for cnt_info2_idx, cnt_info2 in enumerate(self.dilated_chromo_contour_info_list):

                # 自己不同自己算距离
                if cnt_info1_idx == cnt_info2_idx:
                    continue

                # 非常耗时慎用
                # cnt_dist, nearest_pt1, nearest_pt2 = get_min_dist_between_contour(cnt_info1['cnt'], cnt_info2['cnt'])

                # 排除掉一对中的染色体进行合并
                area1 = cnt_info1["area"]
                area2 = cnt_info2["area"]
                area_ratio = area1 / area2 if area1 > area2 else area2 / area1
                if area_ratio < self.AREA_RATIO_BETWEEN_CHROMO_AND_SATE:
                    continue

                # 近似做法，省时间
                cnt_dist, nearest_pt1, nearest_pt2 = get_approx_min_dist_between_rect(
                    cnt_info1["rectCnt"], cnt_info2["rectCnt"]
                )

                if cnt_dist > self.MAX_DISTANCE_BETWEEN_CHROMO_AND_ITS_SATELLITE:
                    continue

                found_close_contours = True
                # 找距离最小的
                if cnt_dist < min_cnt_dist:
                    min_cnt_dist = cnt_dist
                    min_dist_cnt_info1 = cnt_info1
                    min_dist_cnt_info2 = cnt_info2
                    min_dist_cnt_info1_idx = cnt_info1_idx
                    min_dist_cnt_info2_idx = cnt_info2_idx
                    min_dist_nearest_pt1 = nearest_pt1
                    min_dist_nearest_pt2 = nearest_pt2

            if not found_close_contours:
                continue

            merged_cnt = merge_contours(
                self.img.shape,
                min_dist_cnt_info1["cnt"],
                min_dist_cnt_info2["cnt"],
                min_dist_nearest_pt1,
                min_dist_nearest_pt2,
            )

            chromo_and_its_satellite.append(
                {"merged_cnt_idx_list": [min_dist_cnt_info1_idx, min_dist_cnt_info2_idx], "merged_cnt": merged_cnt}
            )

        # 没找到随体
        if not chromo_and_its_satellite:
            return

        # 已经合并的轮廓需要删除
        cnt_for_remove = [idx for cnt_info in chromo_and_its_satellite for idx in cnt_info["merged_cnt_idx_list"]]

        # 去重
        cnt_for_remove = list(set(cnt_for_remove))

        # 删除已经合并的轮廓
        self.dilated_chromo_contour_info_list = [
            cnt_info for idx, cnt_info in enumerate(self.dilated_chromo_contour_info_list) if idx not in cnt_for_remove
        ]

        # 新合并的轮廓列表
        merged_contours = [cnt_info["merged_cnt"] for cnt_info in chromo_and_its_satellite]

        # 从新合并的轮廓中取得轮廓基本信息
        merged_contour_info_list = get_contour_info_list(merged_contours)

        # 将新合并的轮廓添加到轮廓列表中
        self.dilated_chromo_contour_info_list += merged_contour_info_list

        return
        ### END OF merge_chromo_and_its_satellite

    ### 将染色体同编号联系起来
    def link_chromo_with_id(self):

        # 染色体轮廓按行组织
        self.chromo_with_id_rows = [[] for _ in range(len(self.ID_ROW_CY))]
        for i, border in enumerate(self.ID_ROW_ZONE_BORDER_Y):
            self.chromo_with_id_rows[i] = [
                cnt
                for cnt in self.dilated_chromo_contour_info_list
                if cnt["minAR"][0][1] >= border["top"] and cnt["minAR"][0][1] <= border["floor"]
            ]

        # 把染色体编号同垂直膨胀后的染色体轮廓联系起来
        # self.chromo_id_char_xy_info_rows : 染色体编号字符info按行组织
        #
        # 按行循环,计算编号到染色体的距离，最小的就是该染色体所属的编号
        for i, (chromo_list, char_list) in enumerate(zip(self.chromo_with_id_rows, self.chromo_id_char_xy_info_rows)):

            # 对每行的每个染色体求该行各个编号的距离，该染色体属于距离最短的编号
            for chromo_idx, chromo in enumerate(chromo_list):

                chromo_p = chromo["minAR"][0]
                min_distance = self.MAX_DISTANCE
                the_chromo_id = "1"
                the_chromo_num = 1

                for char in char_list:
                    char_p = char["cxy"]
                    the_distance = int(distance(chromo_p, char_p))
                    if the_distance < min_distance:
                        min_distance = the_distance
                        the_chromo_id = char["chromo_id"]
                        the_chromo_num = char["chromo_num"]

                # 220501002.002.0Y.24.00.png
                self.chromo_with_id_rows[i][chromo_idx]["chromo_id"] = the_chromo_id
                self.chromo_with_id_rows[i][chromo_idx]["chromo_num"] = the_chromo_num

        # 每行按染色体编号排序
        for rox_idx in range(len(self.chromo_with_id_rows)):
            self.chromo_with_id_rows[rox_idx] = sorted(self.chromo_with_id_rows[rox_idx], key=itemgetter("chromo_num"))
        ### END of link_chromo_with_id

    ### 打印染色体和编号的对应关系
    def print_chromo_with_id_rows(self):
        for i in range(len(self.chromo_with_id_rows)):
            chromo_list = self.chromo_with_id_rows[i]
            for chromo in chromo_list:
                print(
                    f"{i}:minAR:{chromo['minAR']},area:{chromo['area']},minARarea:{chromo['minARarea']},id:{chromo['chromo_id']},num:{chromo['chromo_num']}"
                )
        ### END of print_chromo_with_id_rows

    ### 根据报告图的轮廓生成单根用于特征提取和匹配的染色体照片
    def save_single_chromo_img_which_cropped_from_rep(self):
        # 将处理后的图片保存到self.chromo_with_id_rows中
        for i in range(len(self.chromo_with_id_rows)):
            for j in range(len(self.chromo_with_id_rows[i])):

                cnt = self.chromo_with_id_rows[i][j]["cnt"]

                self.chromo_with_id_rows[i][j]["chromo_clip_img_from_rep"] = []
                self.chromo_with_id_rows[i][j]["chromo_clip_mask_img_from_rep"] = []

                chromo_clip_img_from_rep, chromo_clip_mask_img_from_rep = get_chromo_img_and_mask_thru_contour_from_rep(
                    cnt, self.img, self.IMG_BIN_THRESH, get_bkg_color(self.img)
                )

                # 原图:0
                self.chromo_with_id_rows[i][j]["chromo_clip_img_from_rep"].append(chromo_clip_img_from_rep)
                self.chromo_with_id_rows[i][j]["chromo_clip_mask_img_from_rep"].append(chromo_clip_mask_img_from_rep)

                # 水平翻转:1
                self.chromo_with_id_rows[i][j]["chromo_clip_img_from_rep"].append(cv2.flip(chromo_clip_img_from_rep, 1))

                self.chromo_with_id_rows[i][j]["chromo_clip_mask_img_from_rep"].append(
                    cv2.flip(chromo_clip_mask_img_from_rep, 1)
                )

                # 垂直翻转:2
                self.chromo_with_id_rows[i][j]["chromo_clip_img_from_rep"].append(cv2.flip(chromo_clip_img_from_rep, 0))

                self.chromo_with_id_rows[i][j]["chromo_clip_mask_img_from_rep"].append(
                    cv2.flip(chromo_clip_mask_img_from_rep, 0)
                )

                # 水平垂直翻转:3
                self.chromo_with_id_rows[i][j]["chromo_clip_img_from_rep"].append(
                    cv2.flip(chromo_clip_img_from_rep, -1)
                )

                self.chromo_with_id_rows[i][j]["chromo_clip_mask_img_from_rep"].append(
                    cv2.flip(chromo_clip_mask_img_from_rep, -1)
                )

        ### END OF save_single_chromo_img


### END of RepImg Class


class OriImg:
    def __init__(self, img_fp):

        if img_fp is None:
            raise ValueError("Original Img init: Img Fullpath is None")

        if not os.path.exists(img_fp):
            raise ValueError(f"Original Img init: image fullpath: {img_fp} does not exist")

        self.fp = img_fp
        (self.fpath, self.fname) = os.path.split(img_fp)
        [self.case_id, self.pic_id, self.f_type, self.f_ext] = self.fname.split(".")
        self.img = cv2.imread(self.fp)

        # Image Properties Init


### END OF OriImg Class


class ChromoMatcher:

    ### 初始化

    global AFFINE_GOOD_MATCHES
    global ROBUST_GOOD_MATCHES
    global CONTRAST_THRESHOLD
    global EDGE_THRESHOLD
    global DIST_THRESHOLD
    global SIGMA
    global PAD

    def __init__(
        self,
        affine_good_matches=AFFINE_GOOD_MATCHES,
        robust_good_matches=ROBUST_GOOD_MATCHES,
        contrast_threshold=CONTRAST_THRESHOLD,
        edge_threshold=EDGE_THRESHOLD,
        dist_threshold=DIST_THRESHOLD,
        sigma=SIGMA,
        pad=PAD,
        debug=False,
    ):

        self.affine_good_matches = affine_good_matches
        self.robust_good_matches = robust_good_matches
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.dist_threshold = dist_threshold
        self.sigma = sigma
        self.pad = pad
        self.debug = debug
        # END of __init__

    ### 染色体同原图做特征匹配
    def _match(
        self,
        chromo_clip,
        ori_img,
        dist_threshold=0.7,
        matcher="bf",
        feature="sift",
        match_box=None,
        dbg=False,
        draw=False,
    ):

        # 使用feature=SIFT, matcher=bf
        det = cv2.xfeatures2d.SIFT_create(
            contrastThreshold=self.contrast_threshold, edgeThreshold=self.edge_threshold, sigma=self.sigma
        )

        # matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matcher = cv2.BFMatcher()

        # 求染色体特征点
        chromo_kp1, chromo_des1 = det.detectAndCompute(chromo_clip, None)

        # 求原图的特征点
        ori_img_kp2, ori_img_des2 = det.detectAndCompute(ori_img, None)

        # match
        matches = matcher.knnMatch(chromo_des1, ori_img_des2, k=2)

        # check result
        good = []
        good_without_list = []
        for m, n in matches:
            if m.distance < self.dist_threshold * n.distance:
                good.append([m])
                good_without_list.append(m)

        result = None

        if dbg:
            result = cv2.drawMatchesKnn(
                chromo_clip, chromo_kp1, ori_img, ori_img_kp2, good, None, flags=cv2.DrawMatchesFlags_DEFAULT
            )

            if draw:
                fig = plt.figure(figsize=(16, 12))
                plt.imshow(result)
                plt.show()

        return {
            "good": good,
            "good_without_list": good_without_list,
            "chromo_kp1": chromo_kp1,
            "ori_img_kp2": ori_img_kp2,
            "draw": result,
        }
        ### END of _match

    ### 求仿射矩阵M
    def _get_affine(self, good_without_list, chromo_kp1, ori_img_kp2):

        src_pts = np.float32([chromo_kp1[m.queryIdx].pt for m in good_without_list]).reshape(-1, 1, 2)
        src_pts -= np.float32([0, 0])

        dst_pts = np.float32([ori_img_kp2[m.trainIdx].pt for m in good_without_list]).reshape(-1, 1, 2)

        affine_m, affine_m_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        return affine_m, affine_m_mask
        ### END of _get_affine

    ### 求当根染色体到原图的仿射矩阵M
    def __get_M(self, match):

        m, m_mask = self._get_affine(match["good_without_list"], match["chromo_kp1"], match["ori_img_kp2"])

        return m, m_mask
        ### END of __get_M

    ### choose best match single chromo image
    def __choose_best_match(self, match_list):

        min_match = self.affine_good_matches

        good_len_list = [len(match["good"]) for match in match_list]

        qualified_good_cnt = sum(good_len >= min_match for good_len in good_len_list)

        if qualified_good_cnt == 0:
            return -1, None, None

        max_cnt = 0
        max_index = 0
        max_m = None
        max_m_mask = None
        for i, good_len in enumerate(good_len_list):
            if good_len >= min_match:
                m, m_mask = self.__get_M(match_list[i])
                mask_count_1 = m_mask.tolist().count([1])
                if mask_count_1 > max_cnt:
                    max_cnt = mask_count_1
                    max_index = i
                    max_m = m
                    max_m_mask = m_mask

        if max_m is None:
            return -1, None, None

        return max_index, max_m, max_m_mask
        ### END of __choose_best_match_img

    ###
    def _warp_single_mask(self, ori_img, m, clip_mask):

        chromo_mask_in_ori = np.zeros_like(ori_img.img)
        h, w = chromo_mask_in_ori.shape[:2]
        try:
            cv2.warpAffine(clip_mask, m, (w, h), chromo_mask_in_ori)
        except Exception as e:
            print(f"_warp_single_mask,{ori_img.fname} error: {e}")
            print(e.args)
            print("=" * 20)
            print(traceback.format_exc())
            raise e

        return chromo_mask_in_ori
        ### END of _warp_single_mask

    ### 匹配报告图做所有染色体，以各种"姿势"(原图、水平、垂直、水平垂直)
    def match(self, ori_img, chromos_in_rows, dbg=False, draw=False, affine=True):

        for i in range(len(chromos_in_rows)):
            for j in range(len(chromos_in_rows[i])):

                # 对染色体的各自姿势就行匹配
                ch_img_list = chromos_in_rows[i][j]["chromo_clip_img_from_rep"]
                ch_img_match_list = [self._match(ch_img, ori_img.img, dbg=dbg, draw=draw) for ch_img in ch_img_list]

                chromos_in_rows[i][j]["match"] = ch_img_match_list

                # choose best match image
                best_match_idx, best_match_m, best_match_m_mask = self.__choose_best_match(ch_img_match_list)

                # save match information
                # if best_match_idx != -1:
                chromos_in_rows[i][j]["best_match_idx"] = best_match_idx
                chromos_in_rows[i][j]["best_match_m"] = best_match_m
                chromos_in_rows[i][j]["best_match_m_mask"] = best_match_m_mask

        return chromos_in_rows
        ### END of match


### END OF ChromoMatcher Class


### DEBUG, 输出报告图中单个染色体的掩码图像
def dbg_save_chromo_clip_img_from_report(output_root_dir, ori_img_fname, chromo_in_rows):

    # G2008311998.124.A.PNG
    [case_id, pic_id, f_type, ext] = ori_img_fname.split(".")
    sub_dir = f"{case_id}.{pic_id}"
    dst_dir = os.path.join(output_root_dir, DBG_DIR, "chromo_mask_from_rep", sub_dir)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for i in range(len(chromo_in_rows)):
        for j in range(len(chromo_in_rows[i])):

            chromo_id = chromo_in_rows[i][j]["chromo_id"].zfill(2)
            chromo_num = str(chromo_in_rows[i][j]["chromo_num"]).zfill(2)
            id2 = str(j).zfill(2)

            for n in range(len(chromo_in_rows[i][j]["chromo_clip_img_from_rep"])):
                ori_img_fname = f"{case_id}.{pic_id}.{chromo_id}.{chromo_num}.{id2}.A{n}.png"
                cv2.imwrite(os.path.join(dst_dir, ori_img_fname), chromo_in_rows[i][j]["chromo_clip_img_from_rep"][n])

            for n in range(len(chromo_in_rows[i][j]["chromo_clip_mask_img_from_rep"])):
                ori_img_fname = f"{case_id}.{pic_id}.{chromo_id}.{chromo_num}.{id2}.M{n}.png"
                cv2.imwrite(
                    os.path.join(dst_dir, ori_img_fname), chromo_in_rows[i][j]["chromo_clip_mask_img_from_rep"][n]
                )
    ### END of dbg_write_chromo_clip_img_from_report


### DEBUG,输出特征匹配图
def dbg_save_chromo_feature_match_img(output_root_dir, fname, chromo_in_rows):

    global DBG_DIR

    # G2008311998.124.A.PNG
    [case_id, pic_id, f_type, ext] = fname.split(".")
    sub_dir = f"{case_id}.{pic_id}"
    dst_dir = os.path.join(output_root_dir, DBG_DIR, "feature_match", sub_dir)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for i in range(len(chromo_in_rows)):
        for j in range(len(chromo_in_rows[i])):

            chromo_id = chromo_in_rows[i][j]["chromo_id"].zfill(2)
            chromo_num = str(chromo_in_rows[i][j]["chromo_num"]).zfill(2)
            id2 = str(j).zfill(2)

            for n in range(len(chromo_in_rows[i][j]["match"])):
                match = chromo_in_rows[i][j]["match"][n]
                fname = f"{case_id}.{pic_id}.{chromo_id}.{chromo_num}.{id2}.FM{n}.png"
                cv2.imwrite(os.path.join(dst_dir, fname), match["draw"])
    ### END of dbg_write_chromo_feature_match_img


### 保存后续处理的基础轮廓和数据
def dbg_save_contours_info_to_file(output_root_dir, rep):
    global DBG_DIR
    canvas = np.zeros(rep.img.shape, dtype=np.uint8)
    d = 1
    for cnt_info in rep.contour_info_list:
        cnt = cnt_info["cnt"]
        # ((cx, cy), (width, height), theta) = cnt_info['minAR']
        (center, size, theta) = cnt_info["minAR"]
        center, size = tuple(map(int, center)), tuple(map(int, size))
        (cx, cy) = center
        (w, h) = size
        cv2.drawContours(canvas, [cnt], -1, (255, 255, 255), 1)
        cv2.putText(
            canvas, f"(x{cx},y{cy})-(w{w},h{h})", (cx + w, cy + (d * 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
        )
        d = 0 - d

    dbg_dir = os.path.join(output_root_dir, DBG_DIR, "bin_img")

    if not os.path.exists(dbg_dir):
        os.makedirs(dbg_dir)
    fname = f"{rep.case_id}.{rep.pic_id}.B.png"
    fp = os.path.join(dbg_dir, fname)
    cv2.imwrite(fp, canvas)
    ### END OF dbg_save_contours_info_to_file


### 单张报告图到原图的染色体匹配
def chromo_matcher_from_rep_to_ori(output_root_dir, ori_fp, rep_fp):

    global DBG_DIR
    global MASK_DIR
    global CHROMO_DIR

    # 载入报告图
    rep_img = RepImg(rep_fp)

    # 根据输入报告图做图片参数校准
    rep_img.img_property_calibration()

    # dbg 保存初始轮廓和信息
    dbg_save_contours_info_to_file(output_root_dir, rep_img)

    # 保存染色体编号字符信息
    rep_img.save_chromo_id_char_xy_info_rows()

    # 保存染色体信息
    rep_img.save_chromo_contour_info_list()

    # 保存垂直膨胀后的染色体信息
    rep_img.save_dilated_chromo_contour_info_list()

    # 合并报告图中分离染色体和其随体
    rep_img.merge_chromo_and_its_satellite()

    # 将染色体同其编号联系起来
    rep_img.link_chromo_with_id()

    # Debug 参看由报告图生成的染色体和对应的编号信息
    # rep_img.print_chromo_with_id_rows()

    # 将报告图中的染色体切割成单个的图像保存(包括水平、垂直、水平垂直翻转后的图像)
    rep_img.save_single_chromo_img_which_cropped_from_rep()
    # self.chromo_with_id_rows[i][j]['chromo_clip_img_from_rep'][0:4] 原图,水平翻,垂直翻,水平垂直翻

    # 载入原图
    ori_img = OriImg(ori_fp)

    # 染色体特征匹配
    matcher = ChromoMatcher()

    ## 将报告图中的每一条染色体同原图进行特征匹配，希冀在原图中找到对应的染色体
    chromos_matcher_in_rows = matcher.match(ori_img, rep_img.chromo_with_id_rows, dbg=False, draw=False)

    # Debug 保存来自报告图的单个染色体图片
    # dbg_save_chromo_clip_img_from_report(output_root_dir, ori_img.fname, chromos_matcher_in_rows)
    # Debug 保存特征匹配图
    # dbg_save_chromo_feature_match_img(output_root_dir, ori_img.fname, chromos_matcher_in_rows)

    # 生成罩回原图的掩码图
    for i in range(len(chromos_matcher_in_rows)):

        # 当前单根染色体在当前染色体编号中的索引
        idx_in_pair = 0
        pre_pair_num = ""
        # pre_pair_num = str(chromos_matcher_in_rows[i][0]['chromo_num']).zfill(2)

        for j in range(len(chromos_matcher_in_rows[i])):
            if chromos_matcher_in_rows[i][j]["best_match_idx"] != -1:

                M = chromos_matcher_in_rows[i][j]["best_match_m"]
                best_match_idx = chromos_matcher_in_rows[i][j]["best_match_idx"]
                chromo_clip_mask_from_rep = chromos_matcher_in_rows[i][j]["chromo_clip_mask_img_from_rep"][
                    best_match_idx
                ]

                # 染色体在原图中的掩码图
                try:
                    chromo_mask_in_ori_img = matcher._warp_single_mask(ori_img, M, chromo_clip_mask_from_rep)
                except Exception as e:
                    print(f"chromo_matcher_from_rep_to_ori : _warp_single_mask : ,{ori_img.fname} error: {e}")
                    print(e.args)
                    print("=" * 20)
                    print(traceback.format_exc())
                    continue

                # 如果掩码图中没有轮廓信息，就表明特征匹配不成功
                # 就不需要保存掩码图了
                if len(find_external_contours(chromo_mask_in_ori_img)) <= 0:
                    chromos_matcher_in_rows[i][j]["chromo_mask_in_ori_img"] = None
                    chromos_matcher_in_rows[i][j]["beautiful_chromo_from_ori_img"] = None
                    continue

                # 在结构中保存染色体在原图中的掩码图
                chromos_matcher_in_rows[i][j]["chromo_mask_in_ori_img"] = chromo_mask_in_ori_img

                # 保存到文件系统
                (fpath, fname) = os.path.split(ori_img.fp)
                # G2008311998.124.A.PNG
                [case_id, pic_id, f_type, ext] = fname.split(".")
                sub_dir = f"{case_id}.{pic_id}"
                mask_dir = os.path.join(output_root_dir, MASK_DIR, sub_dir)
                if not os.path.exists(mask_dir):
                    os.makedirs(mask_dir)
                chromo_id = chromos_matcher_in_rows[i][j]["chromo_id"].zfill(2)
                chromo_num_str = str(chromos_matcher_in_rows[i][j]["chromo_num"]).zfill(2)

                # 对内染色编号
                if chromo_num_str != pre_pair_num:
                    pre_pair_num = chromo_num_str
                    idx_in_pair = 0
                else:
                    idx_in_pair += 1

                # 单根染色体在对内的编号
                id2 = str(idx_in_pair).zfill(2)
                fname = f"{case_id}.{pic_id}.{chromo_id}.{chromo_num_str}.{id2}.png"

                mask_fp = os.path.join(mask_dir, fname)
                # print(dst_fp)
                cv2.imwrite(mask_fp, chromo_mask_in_ori_img)

                # 根据染色体在原图中的掩码图，
                # 在原图中把染色体割出来，
                # 要求：摆正，白底，背景透明
                try:
                    beautiful_chromo_img, chromo_contour_in_ori = crop_img_from_mask(
                        ori_img.img, chromo_mask_in_ori_img, rep_img.IMG_BIN_THRESH
                    )
                except Exception as e:
                    print(f"{ori_img.fp} {e}")
                    print("beautiful_chromo_img error")
                    print(e.args)
                    print("=" * 20)
                    print(traceback.format_exc())
                    continue

                # 单根染色体开口变换和垂直颠倒
                beautiful_chromo_img = chromo_horizontal_flip(beautiful_chromo_img, idx_in_pair)
                beautiful_chromo_img, _ = chromo_stand_up(beautiful_chromo_img)

                # 在结构中保存原图中割出来染色体摆正后的染色体
                chromos_matcher_in_rows[i][j]["beautiful_chromo_from_ori_img"] = beautiful_chromo_img

                chromos_matcher_in_rows[i][j]["chromo_contour_in_ori"] = chromo_contour_in_ori

                # 染色体保存到文件系统
                chromo_dir = os.path.join(output_root_dir, CHROMO_DIR, sub_dir)
                if not os.path.exists(chromo_dir):
                    os.makedirs(chromo_dir)
                chromo_fp = os.path.join(chromo_dir, fname)
                cv2.imwrite(chromo_fp, beautiful_chromo_img)

                # 保存染色体在原图中的轮廓信息
                # numpy.asarray: json to ndarray
                json_fname = f"{os.path.splitext(fname)[0]}.json"
                chromo_contour_fp = os.path.join(mask_dir, json_fname)
                with open(chromo_contour_fp, "w", encoding="utf-8") as f:
                    json.dump(chromo_contour_in_ori.tolist(), f)

    ### END of chromo_matcher_from_rep_to_ori


if __name__ == "__main__":

    # cmd_line: py batch_chromo_matcher.py [ori_dir] [rep_dir] [output_root_dir]

    if len(sys.argv) != 4:
        print(f"Usage: python3 {sys.argv[0]} [ori_dir] [rep_dir] [output_root_dir]")
        exit(1)

    ori_dir = sys.argv[1]
    if not os.path.exists(ori_dir):
        print(f"{ori_dir} not exists")
        exit(2)

    rep_dir = sys.argv[2]
    if not os.path.exists(rep_dir):
        print(f"{rep_dir} not exists")
        exit(3)

    try:
        output_root_dir = sys.argv[3]
        if not os.path.exists(output_root_dir):
            os.makedirs(output_root_dir)

        dbg_root_dir = os.path.join(output_root_dir, DBG_DIR)
        if not os.path.exists(dbg_root_dir):
            os.makedirs(dbg_root_dir)

        output_mask_dir = os.path.join(output_root_dir, MASK_DIR)
        if not os.path.exists(output_mask_dir):
            os.makedirs(output_mask_dir)

        output_chromo_dir = os.path.join(output_root_dir, CHROMO_DIR)
        if not os.path.exists(output_chromo_dir):
            os.makedirs(output_chromo_dir)
    except Exception as e:
        print(f"os.makedirs({sys.argv[3]}) met exception.OR")
        print(f"os.makedirs({dbg_root_dir}) met exception.OR")
        print(f"os.makedirs({output_mask_dir}) met exception.OR")
        print(f"os.makedirs({output_chromo_dir}) met exception.OR")
        print(e.args)
        print("=" * 20)
        print(traceback.format_exc())
        exit(4)

    # 计算工作总量
    img_total = 0
    fnames = os.listdir(rep_dir)
    for fname in fnames:
        [case_id, pic_id, f_type, ext] = fname.split(".")
        if f_type == "K":
            img_total += 1

    t_logger = TimeLogger(img_total)

    for fname in fnames:
        # 220101001.004.K.png
        [case_id, pic_id, f_type, ext] = fname.split(".")
        if f_type != "K":
            continue

        rep_fp = os.path.join(rep_dir, fname)
        ori_fp = os.path.join(ori_dir, f"{case_id}.{pic_id}.A.png")

        t_logger.case_started(fname)
        try:
            chromo_matcher_from_rep_to_ori(output_root_dir, ori_fp, rep_fp)

            t_logger.case_finished(fname)

        except Exception as e:
            print(f"{fname} error: {e}")
            print(e.args)
            print("=" * 20)
            print(traceback.format_exc())

            t_logger.case_finished(fname)
            continue

    t_logger.all_finished()
