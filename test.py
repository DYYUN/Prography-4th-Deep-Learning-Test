# -*- coding: utf-8 -*-
# author: JinTian
# time: 10/05/2017 9:52 AM
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
# File : test.py
# Modified by Daeyoung Yun, At 2019/02/21 5:12 PM
# ------------------------------------------------------------------------
import torch
import os
import sys
from models import DataLoader, fine_tune_model, IMAGE_SIZE, USE_GPU

MODEL_SAVE_FILE = sys.argv[2]

def predict_total_image(inputs, classes_name):
    model = fine_tune_model(classes_name)
    if not os.path.exists(MODEL_SAVE_FILE):
        print('can not find model save file.')
        exit()
    elif USE_GPU:
        model.load_state_dict(torch.load(MODEL_SAVE_FILE))
    else:
        model.load_state_dict(torch.load(MODEL_SAVE_FILE, map_location=lambda storage, loc: storage))

    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(classes_name)))
    class_total = list(0. for i in range(len(classes_name)))
    with torch.no_grad():
        for data in inputs:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            if images.size()[0] is not 1:
                for i in range(images.size()[0]):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    print('Accuracy of the network on the %d test images: %d %%' % (total, (100 * correct / total)))

    for i in range(len(classes_name)):
        print('Accuracy of %5s : %2d %%' % (
            classes_name[i], 100 * class_correct[i] / class_total[i]))

def predict():
    data_loader = DataLoader(data_dir=sys.argv[1], image_size=IMAGE_SIZE)
    predict_total_image(data_loader.load_data(), data_loader.data_classes)

if __name__ == '__main__':
    predict()



