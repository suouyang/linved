import sys

import logging

import os

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append("/home")

import sastvd.helpers.datasets as svdd
import sastvd.ivdetect.evaluate as ivde
import sastvd.helpers.log as log
import coloredlogs

import logging

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")

def bigvul():
    """Run preperation scripts for BigVul dataset."""
    svdd.bigvul()
    # ivde.get_dep_add_lines_bigvul()
    # svdd.generate_glove("bigvul")
    # svdd.generate_d2v("bigvul")


if __name__ == "__main__":
    bigvul()

