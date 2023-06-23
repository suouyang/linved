# Time:2022 2022/3/1 17:20
# Author: Jasmay
# -*- coding: utf-8 -*-

import logging

# 第一步：创建日志器对象，默认等级为warning
logger = logging.getLogger()
logging.basicConfig(level="INFO")

# 第二步：创建控制台日志处理器
console_handler = logging.StreamHandler()

# 第三步：设置控制台日志的输出级别,需要日志器也设置日志级别为info；----根据两个地方的等级进行对比，取日志器的级别
console_handler.setLevel(level="WARNING")

# 第四步：设置控制台日志的输出格式
console_fmt = "%(name)s--->%(asctime)s--->%(message)s--->%(lineno)d"
fmt1 = logging.Formatter(fmt=console_fmt)
console_handler.setFormatter(fmt=fmt1)

# 第五步：将控制台日志器，添加进日志器对象中
logger.addHandler(console_handler)

# print("1/0")

logger.debug("---debug")
# logger.info("---info")
# logger.warning("---warning")
# logger.error("---error")
# logger.critical("---critical")

