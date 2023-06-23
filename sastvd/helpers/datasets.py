import os
import re
import sys
import csv

csv.field_size_limit(sys.maxsize)
import pandas as pd
import sastvd as svd
import sastvd.helpers.doc2vec as svdd2v
import sastvd.helpers.git as svdg
import sastvd.helpers.glove as svdglove
import sastvd.helpers.tokenise as svdt
from sklearn.model_selection import train_test_split
from typing import Sequence
from pandas import Series,DataFrame
# import csv
# csv.QUOTE_NONE
def train_val_test_split_df(df, idcol, labelcol):
    """Add train/val/test column into dataframe."""
    X = df[idcol]
    y = df[labelcol]
    train_rat = 0.8
    val_rat = 0.1
    test_rat = 0.1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_rat, random_state=1
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_rat / (test_rat + val_rat), random_state=1
    )
    X_train = set(X_train)
    X_val = set(X_val)
    X_test = set(X_test)

    def path_to_label(path):
        if path in X_train:
            return "train"
        if path in X_val:
            return "val"
        if path in X_test:
            return "test"

    df["label"] = df[idcol].apply(path_to_label)
    return df


def remove_comments(text):
    """Delete comments from code."""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)
# import sastvd as svd
def bigvul(minimal=False, sample=False, return_raw=False, splits="default"):
    """Read BigVul Data.

    Args:
        sample (bool): Only used for testing!
        splits (str): default, crossproject-(linux|Chrome|Android|qemu)

    EDGE CASE FIXING:
    id = 177860 should not have comments in the before/after
    """
    savedir = svd.get_dir(svd.cache_dir() / "minimal_datasets")

    if minimal:
        try:
            df = pd.read_parquet(
                savedir / f"minimal_bigvul_{sample}.pq", engine="fastparquet"
            ).dropna()

            md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
            # print(md)
            # print(md)
            # print("***********")
            md.groupby("project").count().sort_values("id")

            default_splits = svd.external_dir() / "bigvul_rand_splits.csv"
            if os.path.exists(default_splits):
                splits = pd.read_csv(default_splits)
                splits = splits.set_index("id").to_dict()["label"]
                df["label"] = df.id.map(splits)

            if "crossproject" in splits:
                project = splits.split("_")[-1]
                md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
                nonproject = md[md.project != project].id.tolist()
                trid, vaid = train_test_split(nonproject, test_size=0.1, random_state=1)
                teid = md[md.project == project].id.tolist()
                teid = {k: "test" for k in teid}
                trid = {k: "train" for k in trid}
                vaid = {k: "val" for k in vaid}
                cross_project_splits = {**trid, **vaid, **teid}
                df["label"] = df.id.map(cross_project_splits)

            return df
        except Exception as E:
            print(E)
            pass


    # df['num_words_title'] = df.apply(lambda x: len(x['codeLink'].split('/')), axis=1)
    # print(df)
    # 筛选出影片名长度大于5的部分
    # new_df = df[df['num_words_title'] >= 5]
    # print(df.head(5))
    # print(df.loc['180000','num_words_title'])

    class CsvSequence(Sequence):
        def __init__(self, batchnames):
            self.batchnames = batchnames

        def __len__(self):
            return len(self.batchnames)

        def __getitem__(self, i):
            name = self.batchnames[i]


            X = pd.read_csv("/home/storage/external/"+name+'.csv',engine='python')

                # print(e, type(e))
                # if (isinstance(e, pd.errors.EmptyDataError)):
                #     print("这里对空行文件进行处理")
            return X

    # batchnames = [
       # "MSR_data_cleaned_40001_60001",
    #               "MSR_data_cleaned_40001_60001",
    #                "MSR_data_cleaned_60001_80001_1_5001",
    #               "MSR_data_cleaned_80001_100001",
    #               "MSR_data_cleaned_100001_120001",
    #               "MSR_data_cleaned_120001_140001",
    #               "MSR_data_cleaned_140001_160001",
    #               "MSR_data_cleaned_160001_180001",
    #               "MSR_data_cleaned_180001_188637"
    #               ]
    # csvsequence = CsvSequence(batchnames)
    # a=csvsequence[0]+csvsequence[1]
    # print(a)
    # df = pd.concat([csvsequence[0], csvsequence[1]], ignore_index=True)
    # df= pd.concat([csvsequence[0], csvsequence[1],csvsequence[2],csvsequence[3],csvsequence[4],csvsequence[5],csvsequence[6],csvsequence[7],csvsequence[8]], ignore_index=True)
    # filename = "MSR_data_cleaned_SAMPLE.csv" if sample else "MSR_data_cleaned_180001_188637_1_4001.csv"
    # df = pd.read_csv(svd.external_dir() / filename)
    # print(df)
    # df.to_pickle('/home/storage/external/MSR_mini.pkl')
    # print("**")
    df = DataFrame(pd.read_pickle('/home/storage/external/MSR_5.pkl'))
    print(df)
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["dataset"] = "bigvul"
    # print()
    # print(df.columns)
    # Remove comments
    df["func_before"] = svd.dfmp(df, remove_comments, "func_before", cs=500)

    df["func_after"] = svd.dfmp(df, remove_comments, "func_after", cs=500)
    # print(df)
    # Return raw (for testing)
    if return_raw:
        return df

    # Save codediffs
    cols = ["func_before", "func_after", "id", "dataset"]
    svd.dfmp(df, svdg._c2dhelper, columns=cols, ordr=False, cs=30)

    # Assign info and save
    df["info"] = svd.dfmp(df, svdg.allfunc, cs=500)
    df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)
    # print(df)
    # POST PROCESSING
    # def get_len(row):
    #     print(row['added'])
    #
    # df.apply(get_len, axis=1)
    dfv = df[df.vul == 1]
    # print(dfv.shape)

        # return row['added']

    # dfv.apply(get_len,axis=1)
    # No added or removed but vulnerable
    dfv = dfv[~dfv.apply(lambda x: len(x.added) == 0 and len(x.removed) == 0, axis=1)]
    # print(dfv)
    # Remove functions with abnormal ending (no } or ;)
    # print(dfv.shape)
    # print("***")
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_before.strip()[-1] != "}"
            and x.func_before.strip()[-1] != ";",
            axis=1,
        )
    ]
    # print(dfv)
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_after.strip()[-1] != "}" and x.after.strip()[-1:] != ";",
            axis=1,
        )
    ]
    # print(dfv)
    # Remove functions with abnormal ending (ending with ");")
    # print(dfv)
    dfv = dfv[~dfv.before.apply(lambda x: x[-2:] == ");")]
    # print(dfv.before)

    # Remove samples with mod_prop > 0.5
    # print("***")
    # print(dfv)
    # lambda x: len(x.added + x.removed) / len(x["diff"].splitlines())

    dfv["mod_prop"] = dfv.apply(
        lambda x: len(x.added + x.removed) / len(x["diff"].splitlines()), axis=1
    )

    # def calculate(each_row):
    #    print("*******")
    #    print(each_row.added)
    #    # print(len(each_row['added'])+len(each_row['removed'])/len(each_row['diff'].splitlines()))
    #    return len(each_row['added']+each_row['removed'])/len(each_row["diff"].splitlines())
    #
    # dfv['mod_prop'] = dfv.apply(calculate, axis=1)


    dfv = dfv.sort_values("mod_prop", ascending=0)

    dfv = dfv[dfv.mod_prop < 0.7]
    # Remove functions that are too short
    dfv = dfv[dfv.apply(lambda x: len(x.before.splitlines()) > 5, axis=1)]
    # Filter by post-processing filtering
    keep_vuln = set(dfv.id.tolist())
    df = df[(df.vul == 0) | (df.id.isin(keep_vuln))].copy()

    # Make splits
    df = train_val_test_split_df(df, "id", "vul")


    keepcols = [
        "dataset",
        "id",
        "label",
        "removed",
        "added",
        "diff",
        "before",
        "after",
        "vul",
    ]
    df_savedir = savedir / f"minimal_bigvul_{sample}.pq"
    df[keepcols].to_parquet(
        df_savedir,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )

    metadata_cols = df.columns[:17].tolist() + ["project"]
    # print(metadata_cols)
    df[metadata_cols].to_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv", index=0)
    return df
bigvul()
def generate_glove(dataset="bigvul", sample=False, cache=True):
    """Generate Glove embeddings for tokenised dataset."""
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"glove_{sample}")
    if os.path.exists(savedir / "vectors.txt") and cache:
        svd.debug("Already trained GloVe.")
        return
    if dataset == "bigvul":
        df = bigvul(sample=sample)
    MAX_ITER = 2 if sample else 500
    # print(df)
    # Only train GloVe embeddings on train samples
    samples = df[df.label == "train"].copy()

    # Preprocessing
    samples.before = svd.dfmp(
        samples, svdt.tokenise_lines, "before", cs=200, desc="Get lines: "
    )
    lines = [i for j in samples.before.to_numpy() for i in j]

    # Save corpus
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"glove_{sample}")
    with open(savedir / "corpus.txt", "w") as f:
        f.write("\n".join(lines))

    # Train Glove Model
    CORPUS = savedir / "corpus.txt"
    # print("************")
    svdglove.glove(CORPUS, MAX_ITER=MAX_ITER)
# generate_glove()

def generate_d2v(dataset="bigvul", sample=False, cache=True, **kwargs):
    """Train Doc2Vec model for tokenised dataset."""
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"d2v_{sample}")
    if os.path.exists(savedir / "d2v.model") and cache:
        svd.debug("Already trained Doc2Vec.")
        return
    if dataset == "bigvul":
        df = bigvul(sample=sample)

    # Only train Doc2Vec on train samples
    samples = df[df.label == "train"].copy()

    # Preprocessing
    samples.before = svd.dfmp(
        samples, svdt.tokenise_lines, "before", cs=200, desc="Get lines: "
    )
    lines = [i for j in samples.before.to_numpy() for i in j]
    # print(lines)
    # print("****************")
    # Train Doc2Vec model
    model = svdd2v.train_d2v(lines, **kwargs)

    # Test Most Similar
    # most_sim = model.dv.most_similar([model.infer_vector("memcpy".split())])
    # for i in most_sim:
    #     pass
        # print(lines[i[0]])
    model.save(str(savedir / "d2v.model"))

# generate_d2v()


def bigvul_cve():
    """Return id to cve map."""
    md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
    ret = md[["id", "CVE ID"]]
    return ret.set_index("id").to_dict()["CVE ID"]


# print(bigvul_cve())