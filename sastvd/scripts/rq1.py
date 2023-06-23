import os

import sastvd as svd
import sastvd.linevd.run as lvdrun
from ray import tune

os.environ["SLURM_JOB_NAME"] = "bash"

config = {
    "hfeat": tune.choice([512]),
    "embtype": tune.choice(["codebert", "glove", "doc2vec"]),
    "stmtweight": tune.choice([1, 5, 10]),
    "hdropout": tune.choice([0.1, 0.15, 0.2, 0.25]),
    "gatdropout": tune.choice([0.15, 0.2]),
    "modeltype": tune.choice(["mlponly"]),
    "gnntype": tune.choice(["gat"]),
    "loss": tune.choice(["ce"]),
    "scea": tune.choice([0.4, 0.5, 0.6]),
    "gtype": tune.choice(["pdg+raw"]),
    "batch_size": tune.choice([1024]),
    "multitask": tune.choice(["line"]),
    "splits": tune.choice(["default"]),
    "lr": tune.choice([1e-3, 1e-4, 3e-4, 5e-4]),
}

samplesz = -1
run_id = svd.get_run_id()
sp = svd.get_dir(svd.processed_dir() / f"raytune_features_{samplesz}" / run_id)
trainable = tune.with_parameters(lvdrun.train_linevd, samplesz=samplesz, savepath=sp)

analysis = tune.run(
    trainable,
    resources_per_trial={"cpu": 1, "gpu": 1},
    metric="val_loss",
    mode="min",
    config=config,
    num_samples=1000,
    name="tune_linevd",
    local_dir=sp,
    keep_checkpoints_num=2,
    checkpoint_score_attr="min-val_loss",
)
