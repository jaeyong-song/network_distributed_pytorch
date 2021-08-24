import datetime
import os

import numpy as np
from math import ceil

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, AdamW

from torch.autograd import Variable

import partition_helper as part_help
from reducer import PowerSGDReducer

config = dict (
    seed=714,
    rank=0, # should be updated by caller
    cuda_rank=0,
    n_workers=4,
    distributed_init_file=None, #(Kind of Socket)
    output_dir="./output.tmp", # you must create this directory
    distributed_backend="nccl", # gloo is more compatible recently
    init_method="tcp://165.132.142.56:7392",

    learning_rate = 5e-5,
    momentum = 0.9,
    nesterov = False,
    training_epochs = 5,
    batch_size = 16,
    reducer_rank = 16,
)


# For DistillBERT IMDb Reviews
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels


def prepare_IMDb():
    train_texts, train_labels = read_imdb_split('/home/seonbinara/aclImdb/train')
    test_texts, test_labels = read_imdb_split('/home/seonbinara/aclImdb/test')

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)
    # return (train_set, bsz), val_set, test_set
    return partition_dataset(train_dataset), val_dataset, test_dataset

def partition_dataset(dataset):
    """Partitioning IMDb"""
    size = dist.get_world_size()
    # just use 16 is ok...
    total_batch = 16 * size
    bsz = int(total_batch / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = part_help.DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    return partition, bsz

def average_gradients(model):
    """Gradient Averaging"""
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def setup():
    torch.manual_seed(config["seed"] + config["rank"])
    np.random.seed(config["seed"] + config["rank"])

    device = torch.device("cuda:" + str(config["cuda_rank"]) if torch.cuda.is_available() else "cpu")

    # PyTorch Distributed Training Init
    # By. init_process_group
    if dist.is_available():
        print("==============================")
        print(">>>>> PyTorch DDP Initialization Step <<<<<")
        if config["distributed_init_file"] is None:
            config["distributed_init_file"] = os.path.join(config["output_dir"], "dist_init")
        print(
            "Distributed Init: rank {}/{}(Total: {}) - socket ({})".format(
                config["rank"], config["n_workers"]-1, config["n_workers"], config["init_method"] 
            )
        )
        dist.init_process_group(
            backend=config["distributed_backend"],
            # By using nfs... use shared file-system initialization
            # If not using TCP initialization
            # Our Lab nfs is not fast....
            # Use Direct Connection
            # for internal socket use file
            # for external socekt use tcp connection
            # init_method="file://" + os.path.abspath(config["distributed_init_file"]),
            init_method=config["init_method"],
            timeout=datetime.timedelta(seconds=600),
            world_size=config["n_workers"],
            rank=config["rank"]
        )
        print("All ranks successfully initialized")
        print("==============================\n")
    else:
        print("[Failure] Distributed Environment Failed")

def run_task():
    print("==============================")
    print(">>>>> Run Designated Task <<<<<")

    torch.manual_seed(config["seed"] + config["rank"])
    np.random.seed(config["seed"] + config["rank"])

    device = torch.device("cuda:" + str(config["cuda_rank"]) if torch.cuda.is_available() else "cpu")
    (train_set, bsz), val_set, test_set = prepare_IMDb()

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)

    model.train()

    train_loader = DataLoader(train_set, batch_size=bsz, shuffle=True)

    # Do not use momentum or nesterov...
    # It is self implemented below
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

    num_batches = ceil(len(train_loader.dataset) / float(bsz))

    # Reducer for PowerSGD
    reducer = PowerSGDReducer(random_seed=config["seed"], device=device, n_power_iterations=0, reuse_query=True, rank=config["reducer_rank"])

    bits_communicated = 0

    """
    PowerSGD
    Algorithm 2: Distributed Error-feedback SGD with Momentum
    """
    # (Algo 2: line 4) initialize memory e
    state = [parameter for parameter in model.parameters()]
    memories = [torch.zeros_like(param) for param in state]
    # Make Temp Buffer for intermediate calculation
    send_buffers = [torch.zeros_like(param) for param in state]
    # Need Initialization for Momentum Calculation
    momenta = [torch.empty_like(param) for param in state]

    # (Algo 2: line 5) for each iterate t = 0, ... do
    for epoch in range(config["training_epochs"]):
        print(">>>>> Rank ", dist.get_rank(), ", epoch ", epoch, " Started...")
        epoch_loss = 0.0
        i = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            epoch_loss += loss.item()

            # (Algo 2: line 6) Compute a SG g
            loss.backward()
            grads = [parameter.grad for parameter in model.parameters()]
            
            # !!!!! PowerSGD Logic Should be here !!!!!
            
            # (Algo 2: line 7) delta_w = g + e (save to send_buffers)
            for g, e, send_buffer in zip(grads, memories, send_buffers):
                send_buffer.data[:] = g + e
            
            # (Algo 2: line 8-11) Compress/Decompress and Memorize Local Errors
            # Implemented in Reducer
            bits_communicated += reducer.reduce(send_buffers, grads, memories)
            # Calculated output is in grads

            # [CF] Use Common Momentum Equation
            # (Algo 2: line 12) m <- \lambda * m + decompressed_gradient(delta)
            for g, momentum in zip(grads, momenta):
                if epoch == 0 and i == 0:
                    momentum.data = g.clone().detach()
                else:
                    momentum.mul_(config["momentum"]).add_(g)
                # (Algo 2: line 13) (grad + momentum)
                g.data[:] += momentum
            
            # Use Custom Step
            # [Do not use] optimizer.step()
            # (Algo 2: line 13) x <- x - \gamma*(grad + momentum)
            for param, g in zip(model.parameters(), grads):
                param.data.add_(g, alpha=-config["learning_rate"])
            
            # for small batch counting
            i += 1

        print("     Rank ", dist.get_rank(), ", epoch ", epoch, ": ", epoch_loss / num_batches)
        print(">>>>> Rank ", dist.get_rank(), ", epoch ", epoch, " Finished...\n")

    print("All Task Finished")
    print("==============================\n")

def cleanup():
    print("==============================")
    print(">>>>> PyTorch DDP Destroy <<<<<")
    dist.destroy_process_group()
    print("All ranks successfully destroyed")
    print("==============================\n")

if __name__ == "__main__":
    setup()
    run_task()
    cleanup()
        
