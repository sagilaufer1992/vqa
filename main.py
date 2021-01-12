import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image

import os
from PIL import Image
import torch.optim as optim

import matplotlib.pyplot as plt

from vqaDataset import VqaDataset
from vqaModule import VqaModule
from vqaTrainer import VqaTrainer
import pycuda.driver as cuda
import copy

from tqdm import tqdm

batch_size = 300


def save_models(epochs, model):
    torch.save(model.state_dict(), "custom_model{}.model".format(epochs))
    print("Checkpoint Saved")


def train_model(model, criterion, optimizer, scheduler=None, num_epochs=250):
    print("train len:", len(train_dataset))
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for sample in tqdm(train_loader):
            sample["image"], sample["question"], sample["label"] = sample["image"].cuda(), sample["question"].cuda(), sample["label"].cuda()
            # zero the parameter gradients
            # print(sample["label"])
            optimizer.zero_grad()

            # forward
            # track history if only in train
            outputs = model(sample["image"], sample["question"])
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, torch.argmax(sample["label"], dim=1).cuda())
            # print(loss)
                # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            #statistics
            running_loss += loss.item() * batch_size

            # zeros_mask = torch.sum(sample["label"], dim=1) != 0
            # print(zeros_mask, sum(zeros_mask), len(zeros_mask))
            # print(preds, outputs)
            correct_mask = preds == torch.argmax(sample["label"], dim=1)
            # print(correct_mask, sum(correct_mask), len(correct_mask))
            # total_mask = torch.logical_and(zeros_mask, correct_mask)

            running_corrects += torch.sum(correct_mask, dim=0)

        # scheduler.step()

      

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset) * 100

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    #     # deep copy the model
    #     if phase == 'train' and epoch_acc > best_acc:
        save_models(epoch, model)
        # best_acc = epoch_acc
        # best_model_wts = copy.deepcopy(model.state_dict())

    print()   

    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # return model










cuda.init()

train_dataset = VqaDataset(images_path="./vqa_data/train/images/train2014/", 
                           questions_path="./vqa_data/train/v2_OpenEnded_mscoco_train2014_questions.json", 
                           annotations_path="./vqa_data/train/v2_mscoco_train2014_annotations.json")

train_loader = DataLoader(train_dataset, batch_size, True)
mod = VqaModule()
mod.cuda()
criterion = nn.CrossEntropyLoss(train_dataset.answers_weights)
# optimizer_ft = optim.Adagrad(mod.parameters(), lr=0.01, lr_decay=0.1)
optimizer_ft = optim.SGD(mod.parameters(), lr=0.05, momentum=0.90)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

train_model(mod, criterion, optimizer_ft, num_epochs=250)



