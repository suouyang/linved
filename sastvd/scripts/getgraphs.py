import os
import pickle as pkl
import sys
import sys

import logging

import os

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append("/home")

import numpy as np
import sastvd as svd
import  pandas as pd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.joern as svdj
import sastvd.helpers.sast as sast

# SETUP
NUM_JOBS = 10
JOB_ARRAY_NUMBER = 0 if "ipykernel" in sys.argv[0] else int(sys.argv[1]) - 1
# print(JOB_ARRAY_NUMBER)
# Read Data
# df = svdd.bigvul()
# df = df.iloc[::-1]
df=pd.read_parquet("/home/storage/cache/minimal_datasets/minimal_bigvul_False.pq")
# print(df)
# def cal(row):
#     # if(len(row['diff'])>0):
#         # print(row['id'])
# df.apply(cal,axis=1)
# splits = np.array_split(df, NUM_JOBS)
# print(splits)
# print("*************")
def preprocess(row):
    """Parallelise svdj functions.

    Example:
    df = svdd.bigvul()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    savedir_before = svd.get_dir(svd.processed_dir() / row["dataset"] / "before")
    savedir_after = svd.get_dir(svd.processed_dir() / row["dataset"] / "after")
    #
    # # Write C Files
    fpath1 = savedir_before / f"{row['id']}.c"
    # # # count=0
    # with open(fpath1, "w") as f:
    #
    #     f.write(row["before"])
    # # #
    fpath2 = savedir_after / f"{row['id']}.c"
    # # # print(len(row["diff"]))
    # if len(row['diff']) > 0:
    # #     count = count + 1
    #     with open(fpath2, "w") as f:
    #         f.write(row["after"])
    # print(count)
    # Run Joern on "before" code
    if not os.path.exists(f"{fpath1}.edges.json"):
        svdj.full_run_joern(fpath1, verbose=3)
    # print("*****************")
    # Run Joern on "after" code
    # if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
    #     svdj.full_run_joern_after(fpath2, verbose=3)
    #
    # # Run SAST extraction
    # fpath3 = savedir_before / f"{row['id']}.c.sast.pkl"
    # if not os.path.exists(fpath3):
    #     sast_before = sast.run_sast(row["before"])
    #     with open(fpath3, "wb") as f:
    #         pkl.dump(sast_before, f)
# preprocess()

if __name__ == "__main__":
    svd.dfmp(df, preprocess, ordr=False, workers=8)
# df.apply(preprocess,axis=1)