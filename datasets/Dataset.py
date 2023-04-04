
import os
import numpy as np
import cv2
 
# 导入PyTorch库
import torch
from torch.utils.data import Dataset

class EtEN_Datasets(Dataset):
    def __init__(self, data_path:str, model:str) -> None:
        super().__init__()

        self.path = data_path
        self.model = model        
        assert self.mode in {'train', 'val'}
        self.data_info = self. get_data_info

    def get_data_info(self):
        file = os.path.join(self.path, self.model, "_info.txt")
        f = open(file)
        list = eval(f.readline())
        return list


    def __getitem__(self, i):
        """
        : i: the image index
        : return the image of the index i
        """
        # read image
        img = cv2.imread(self.data_info[i]["path"])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        return img, i

    def __len__(self):
        return len(self.data_info)