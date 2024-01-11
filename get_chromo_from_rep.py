"""从报告图中抠取染色体图像
"""
import configparser
from utils.chromo_cv_utils import find_external_contours


cfg = configparser.ConfigParser()
cfg.read("get_chromo_from_rep.ini", encoding="utf-8")

chromo_id_row_heights = [
    int(height.strip()) for height in cfg["General"]["chromo_id_row_heights"].split(",")
]

