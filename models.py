#!/usr/bin/env Python
# coding=utf-8

import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc_encode.in_features +
                                  self.fc_encode.out_features)
        self.fc_encode.weight.data.uniform_(-r, r)
        self.fc_encode.bias.data.fill_(0)

    def forward(self, x):
        hid = self.fc_encode(x)
        code = torch.tanh(self.alpha * hid)

        return hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.alpha = 1.0
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc1.in_features +
                                  self.fc1.out_features)
        self.fc1.weight.data.uniform_(-r, r)
        self.fc1.bias.data.fill_(0)

        r = np.sqrt(6.) / np.sqrt(self.fc2.in_features +
                                  self.fc2.out_features)
        self.fc2.weight.data.uniform_(-r, r)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = torch.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
