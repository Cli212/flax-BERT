import subprocess
import argparse
import csv
import ast
from tqdm import tqdm
import time
import os
import pandas as pd
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--layout", default="NT", type=str)
    parser.add_argument("--train", default=True, type=ast.literal_eval)
    parser.add_argument("--fp16", default=False, type=ast.literal_eval)
    parser.add_argument("--csv_file",default="train_speed_memory.csv",type=str)
    args = parser.parse_args()
    layout = args.layout
    repeat = args.repeat
    fp16 = args.fp16
    csv_file = args.csv_file
    file = args.csv_file
    workload_dict = {"train": [(4, 128),
                               (8, 128),
                               (16, 128),
                               (32, 128),
                               (1, 512),
                               (2, 512),
                               (4, 512),
                               (8, 512)
                               ], "infer": [(1, 128),
                                            (1, 384),
                                            (1, 512),
                                            (8, 32),
                                            (8, 128),
                                            (8, 512),
                                            (32, 512),
                                            (256, 128),
                                            (400, 100)
                                            ]}
    models = ["bert-base-uncased", "bert-large-uncased"]
    workloads = workload_dict["train"] if args.train else workload_dict["infer"]
    for model_name in models:
        for workload in tqdm(workloads):
            if layout == "NT":
                batch_size, seq_length = workload
            elif layout == "TN":
                seq_length, batch_size = workload
            else:
                raise ValueError("Invalid Layout")
            cmd = ["python","speed_test.py","--train",f"{args.train}","--model_name",model_name,"--batch_size",f"{batch_size}","--seq_length",f"{seq_length}","--fp16",f"{fp16}","--repeat",f"{repeat}","--layout",layout,"--file",csv_file]
            p = subprocess.Popen(cmd, stdin = subprocess.PIPE, stdout=subprocess.PIPE)
            out = p.stdout.readlines()
            p.kill()
    df_list = []
    for file in os.listdir(f"{'train' if args.train else 'infer'}"):
        df = pd.read_csv(os.path.join(f"{'train' if args.train else 'infer'}",file))
        df_list.append(df)
    ddf = pd.concat(df_list,axis=0)
    ddf.sort_values(['model_name', 'batch_size', 'sequence_length'], inplace=True)
    ddf.to_csv(csv_file)
