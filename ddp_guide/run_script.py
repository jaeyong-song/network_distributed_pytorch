import ddp_init
import argparse

"""
PowerSGD Execution Code
Execute for all nodes(ranks)
or you can execute by mpirun
>>>> if use mpirun
    # test.py
    import os
    import train
    train.config["num_epochs"] = 200
    train.config["n_workers"] = os.gettenv("OMPI_COMM_WORLD_SIZE")
    train.config["rank"] = os.gettenv("OMPI_COMM_WORLD_RANK")
    train.config["distributed_init_file"] = "some unique file on a file system workers can all access"

    # other configuration like setting output_dir, and logging functions
    # ...

    train.main()
<<<<<
then, $ mpirun -n 4 python3 test.py
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rank", type=int, required=True, help="worker id")
    parser.add_argument("-cuda", type=int, required=True, help="worker's cuda id")
    args = parser.parse_args()

    # import config and set custom configs

    # Configure the worker
    ddp_init.config["n_workers"] = 4
    ddp_init.config["rank"] = args.rank # number of this worker in [0,4).
    ddp_init.config["cuda_rnak"] = args.cuda

    # Start training
    ddp_init.main()