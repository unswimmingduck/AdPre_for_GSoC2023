
import os
import argparse
from pathlib import Path
import random
 


def creat_data_info(root_path:str,
                    model:str):
    path = os.path.join(Path(root_path), model)
    data_lsit = os.listdir(path)
    info = []
    for data in data_lsit:
        detail = {}
        detail['path'] = os.path.join(path, data)
        detail['angular'] = data.split(".")[-1]
        info.append(detail)
    f=open(path+ model + "_info.txt","w")
    f.write(info)
    f.close()

def main():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument('--root-path',
                        type=str,
                        default='./data',
                        help='specify the root path of dataset')                    
    args = parser.parse_args()
    root_path = args.root_path
    creat_data_info(root_path, "train")
    creat_data_info(root_path, "val")


