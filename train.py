# -*- coding: utf-8 -*-
# author: JinTian
# time: 10/05/2017 8:52 AM
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
# File : train.py
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
import sys
from models import DataLoader, save_torch_model, train_model, exp_lr_scheduler, fine_tune_model, IMAGE_SIZE, USE_GPU

MODEL_SAVE_FILE = 'model_new.pth'

def train():

    data_loader = DataLoader(data_dir=sys.argv[1], image_size=IMAGE_SIZE, batch_size=4)
    inputs, classes = next(iter(data_loader.load_data()))
    out = torchvision.utils.make_grid(inputs)
    data_loader.show_image(out, title=[data_loader.data_classes[c] for c in classes])

    model = fine_tune_model(data_loader.data_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    try:
        model = train_model(data_loader, model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
        save_torch_model(model, MODEL_SAVE_FILE)
    except KeyboardInterrupt:
        print('manually interrupt, try saving model for now...')
        save_torch_model(model, MODEL_SAVE_FILE)
        print('model saved.')


def main():
    train()

if __name__ == '__main__':
    main()