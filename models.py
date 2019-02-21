# -*- coding: utf-8 -*-
# author: JinTian
# time: 10/05/2017 9:54 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
# File : models.py
# Modified by Daeyoung Yun, At 2019/02/21 5:12 PM
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import Image

IMAGE_SIZE = 224
USE_GPU = torch.cuda.is_available()

class DataLoader(object):
    def __init__(self, data_dir, image_size, batch_size=4):
        """
        this class is the normalize data loader of PyTorch.
        The target image size and transforms can edit here.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.normalize_mean, self.normalize_std)
            ])


        self._init_data_sets()

    def _init_data_sets(self):
        self.data_sets = datasets.ImageFolder(root=self.data_dir, transform=self.data_transforms)
        self.data_loaders = torch.utils.data.DataLoader(self.data_sets, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.data_sizes = len(self.data_sets)
        self.data_classes = self.data_sets.classes

    def load_data(self):
        return self.data_loaders

    def show_image(self, tensor, title=None):
        inp = tensor.numpy().transpose((1, 2, 0))
        # put it back as it solved before in transforms
        inp = self.normalize_std * inp + self.normalize_mean
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.show()

def save_torch_model(model, name):
    torch.save(model.state_dict(), name)


def train_model(data_loader, model, criterion, optimizer, lr_scheduler, num_epochs=25):
    """
    the pipeline of train PyTorch model
    :param data_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param lr_scheduler:
    :param num_epochs:
    :return:
    """
    since_time = time.time()

    best_model = model

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        optimizer = lr_scheduler(optimizer, epoch)
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for batch_data in data_loader.load_data():
            inputs, labels = batch_data
            if USE_GPU:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

                # collect data info
            running_loss += loss.item()
            running_corrects += (predict == labels).sum().item()

        epoch_loss = running_loss / data_loader.data_sizes
        epoch_acc = running_corrects / data_loader.data_sizes

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(
                epoch_loss, epoch_acc))

        print()

    time_elapsed = time.time() - since_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_model


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def fine_tune_model(classes_name):
    model_ft = models.resnet18(pretrained=True)
    num_features = model_ft.fc.in_features
    # fine tune we change original fc layer into classes num of our own
    model_ft.fc = nn.Linear(num_features, len(classes_name))

    if USE_GPU:
        model_ft = model_ft.cuda()
    return model_ft