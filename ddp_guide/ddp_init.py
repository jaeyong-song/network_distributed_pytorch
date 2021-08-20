import datetime
import os
import re
import time

import torch
import numpy as np

config = dict (
    seed=714,
    rank=0, # should be updated by caller
    cuda_rank=0,
    n_workers=4,
    distributed_init_file=None, #(Kind of Socket)
    output_dir="./output.tmp", # you must create this directory
    distributed_backend="nccl", # gloo is more compatible recently
)

def main():
    torch.manual_seed(config["seed"] + config["rank"])
    np.random.seed(config["seed"] + config["rank"])

    device = torch.device("cuda:" + str(config["rank"]) if torch.cuda.is_available() else "cpu")

    # PyTorch Distributed Training Init
    # By. init_process_group
    if torch.distributed.is_available():
        print("==============================")
        print(">>>>> PyTorch DDP Initialization Step <<<<<")
        if config["distributed_init_file"] is None:
            config["distributed_init_file"] = os.path.join(config["output_dir"], "dist_init")
        print(
            "Distributed Init: rank {}/{}(Total: {}) - socket ({})".format(
                config["rank"], config["n_workers"]-1, config["n_workers"], config["distributed_init_file"] 
            )
        )
        torch.distributed.init_process_group(
            backend=config["distributed_backend"],
            # By using nfs... use shared file-system initialization
            # If not using TCP initialization
            init_method="file://" + os.path.abspath(config["distributed_init_file"]),
            timeout=datetime.timedelta(seconds=120),
            world_size=config["n_workers"],
            rank=config["rank"]
        )
        print("All ranks successfully initialized")
        print("==============================\n")

if __name__ == "__main__":
    main()
        
