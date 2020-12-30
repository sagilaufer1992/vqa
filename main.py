import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image

import os
from PIL import Image

import matplotlib.pyplot as plt

from vqaDataset import VqaDataset
from vqaModule import VqaModule


train_dataset = VqaDataset(images_path="./vqa_data/train/images/train2014/", 
                           questions_path="./vqa_data/train/v2_OpenEnded_mscoco_train2014_questions.json", 
                           annotations_path="./vqa_data/train/v2_mscoco_train2014_annotations.json")

train_loader = DataLoader(train_dataset, 30, True)

mod = VqaModule()

for i, (im, a, q) in enumerate(train_loader):
    # print (a)
    # save_image(im[0], "img1.png")
    mod.forward(im, q)
    break

