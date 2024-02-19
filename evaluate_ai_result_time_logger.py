# -*- coding: utf-8 -*-
"""记录程序运行时间,估计程序运行时间

Usage:
    - Import this module using `import mymodule`.
    - Use the functions provided by this module as needed.

Author: Lei Wang
Date: Feb 5, 2024
"""

__author__ = "王磊"
__copyright__ = "Copyright 2023 四川科莫生医疗科技有限公司"
__credits__ = ["王磊"]
__maintainer__ = "王磊"
__email__ = "lei.wang@kemoshen.com"
__version__ = "0.0.1"
__status__ = "Development"


import time
from evaluate_ai_result_logger import log

NEW_LINE = "\n"


class TimeLogger:
    """Used to log process time of every step
    """
    def __init__(self, outside_logger=log, case_total_cnt=0):
        self.st = time.time()
        self.case_total = case_total_cnt
        self.processed = 0
        self.avg_cost = 1
        self.case_st = 0
        self.case_et = 0
        self.et = 0
        self.logger = outside_logger

    def case_started(self, cur_case):
        """log start time of the case

        Args:
            cur_case (_type_): _description_
        """
        self.case_st = time.time()
        self.estimate_cur_case_end_time(cur_case, " started, estimated end on: ")

    def case_finished(self, cur_case):
        """log end time of the case

        Args:
            cur_case (_type_): _description_
        """
        self.processed += 1
        self.case_et = time.time()
        self.avg_cost = (self.case_et - self.st) / self.processed

        local_t = time.localtime(self.case_et)
        f_local_t = time.strftime("%Y-%m-%d %H:%M:%S", local_t)

        self.logger.info(
            f"{NEW_LINE}{cur_case} finished {f_local_t}, cost: {self.case_et - self.case_st:.2f}s, avg: {self.avg_cost:.2f}s, T{self.case_total}-P{self.processed}-L{self.case_total-self.processed}."
        )

        self.estimate_end_time("End of all cases estimated on: ")

    def all_finished(self):
        """calculate the average cost of all cases and log the end time of all cases
        """
        self.et = time.time()
        self.avg_cost = (self.et - self.st) / self.processed

        local_t = time.localtime(self.et)
        f_local_t = time.strftime("%Y-%m-%d %H:%M:%S", local_t)

        self.logger.info(f"{NEW_LINE}" + "=" * 20)
        self.logger.info(
            f"{NEW_LINE}All cases finished on {f_local_t}, cost: {self.et - self.st:.2f}s, avg: {self.avg_cost:.2f}s."
        )

    def estimate_cur_case_end_time(self, cur_case, arg1):
        """estimate the end time of the current case

        Args:
            cur_case (_type_): _description_
            arg1 (_type_): _description_
        """
        estimated_end_time = self.case_st + self.avg_cost

        local_t = time.localtime(estimated_end_time)
        f_local_t = time.strftime("%Y-%m-%d %H:%M:%S", local_t)

        self.logger.info(f"{NEW_LINE}{cur_case}{arg1}{f_local_t}")

    def estimate_end_time(self, arg1):
        """estimate the end time of all cases

        Args:
            arg1 (_type_): _description_
        """
        estimated_end_time = self.case_et + (
            self.avg_cost * (self.case_total - self.processed)
        )

        local_t = time.localtime(estimated_end_time)
        f_local_t = time.strftime("%Y-%m-%d %H:%M:%S", local_t)

        self.logger.info(f"{NEW_LINE}{arg1}{f_local_t}")
