# -*- coding: utf-8 -*-
# @Time : 2023/2/2 13:46
# @Author : zxf
import os
import logging


def get_logger(name, log_file=None, log_level='DEBUG'):
    """
    logger
    :param name: 模块名称
    :param log_file: 日志文件，如无则输出到标准输出
    :param log_level: 日志级别
    :return:
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(log_level.upper())
    formatter = logging.Formatter('[%(levelname)7s %(asctime)s %(module)s:%(lineno)4d] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
                                  # datefmt='%Y%m%d %I:%M:%S')

    if log_file:
        dirname = os.path.dirname(log_file)
        os.makedirs(dirname, exist_ok=True)
        f_handle = logging.FileHandler(log_file)
        f_handle.setFormatter(formatter)
        logger.addHandler(f_handle)
    handle = logging.StreamHandler()
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    return logger


logger = get_logger(__name__, log_file=None, log_level='DEBUG')


def set_log_level(log_level='INFO'):
    logger.setLevel(log_level.upper())