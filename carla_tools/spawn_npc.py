#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import time
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.append('../carla/')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
import weakref
import argparse
import logging
import random, math
from queue import Queue
from queue import Empty
from functools import partial
from agents.tools.misc import draw_waypoints, get_speed
from basic_agent import BasicAgent  # pylint: disable=import-error
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../FedML_v4/fedml_experiments/standalone/decentralized_pred/')
from models import *

# ======================================================================================================================
# -- Global variables. -------------------------------------------------------------------------------------
# ======================================================================================================================
'''
class CONV_NN_v3_1_agent_4_view(nn.Module):
    # frame prediction
    def __init__(self, num_classes, input_dim, h_units, act, args):
        super(CONV_NN_v3_1_agent_4_view, self).__init__()
        # [10, 128, 128, 5]
        self.filter_size = args.filter_size
        self.padding = self.filter_size // 2

        self.frame_predictor = DeterministicConvLSTM(input_dim, h_units[0], h_units[0], 1, args)
        self.l3_1 = nn.Conv3d(h_units[0], h_units[1], kernel_size=self.filter_size, stride=1,
                            padding=self.padding, bias=False)
        self.bn3d_1 = nn.BatchNorm3d(h_units[1])
        self.l3_2 = nn.Conv3d(h_units[1], h_units[2], kernel_size=self.filter_size, stride=1,
                            padding=self.padding, bias=False)
        self.bn3d_2 = nn.BatchNorm3d(h_units[2])
        self.l3_3 = nn.Conv3d(h_units[2], h_units[3], kernel_size=self.filter_size, stride=1,
                            padding=self.padding, bias=False)

        self.l3 = nn.Conv3d(h_units[-1], num_classes*args.num_views, kernel_size=self.filter_size, stride=1, padding=self.padding, bias=False)

        if act == "relu":
            self.act = F.relu
        elif act == "sig":
            self.act = F.sigmoid

    def __call__(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1, message, memory #, h_t, c_t, delta_c_list, delta_m_list

    def forward(self, x_t, m_t, m_t_others, frame_predictor_hidden):
        x = torch.cat([x_t, ], -1)
        x = x.permute(0, 4, 1, 2, 3)
        h, frame_predictor_hidden = self.frame_predictor(x, frame_predictor_hidden)
        h = self.l3_1(h)
        h = self.act(h)
        h = self.bn3d_1(h)
        h = self.l3_2(h)
        h = self.act(h)
        h = self.bn3d_2(h)
        h = self.l3_3(h)
        h = self.act(h)

        pred_x_tp1 = self.l3(h)
        message = None
        pred_x_tp1 = pred_x_tp1.permute(0, 2, 3, 4, 1)
        return pred_x_tp1, message, frame_predictor_hidden  # h_t, c_t, delta_c_list, delta_m_list

    def predict(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1.data, message.data, memory #, h_t, c_t, delta_c_list, delta_m_list
'''

class CONV_NN_v3_1_agent_2_view(nn.Module):
    # frame prediction
    def __init__(self, input_dim, h_units, act, args):
        super(CONV_NN_v3_1_agent_2_view, self).__init__()
        # [10, 128, 128, 5]
        self.filter_size = args.filter_size
        self.padding = self.filter_size // 2

        self.frame_predictor = DeterministicConvLSTM(input_dim, h_units[-1], h_units[0], len(h_units), args)
        self.l3 = nn.Conv3d(h_units[-1], 3*args.num_views, kernel_size=self.filter_size, stride=1, padding=self.padding, bias=False)

        if act == "relu":
            self.act = F.relu
        elif act == "sig":
            self.act = F.sigmoid

    def __call__(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1, message, memory #, h_t, c_t, delta_c_list, delta_m_list

    def forward(self, x_t, m_t, m_t_others, frame_predictor_hidden):
        # print('x_t: {}, m_t: {}, m_t_others: {}'.format(x_t.shape, m_t.shape, m_t_others.shape))  # [10, 3, 128, 128, 3], [10, 1, 128, 128, 1], [10, 1, 128, 128, 1]
        # m_t = m_t.repeat(1, x_t.shape[1] // m_t.shape[1], 1, 1, 1)
        # m_t_others = m_t_others.repeat(1, x_t.shape[1] // m_t_others.shape[1], 1, 1, 1)
        # x = torch.cat([x_t, m_t, m_t_others, ], -1)
        x = torch.cat([x_t, ], -1)
        x = x.permute(0, 4, 1, 2, 3)
        h, frame_predictor_hidden = self.frame_predictor(x, frame_predictor_hidden)
        pred_x_tp1 = self.l3(h)
        message = None
        pred_x_tp1 = pred_x_tp1.permute(0, 2, 3, 4, 1)
        pred_x_tp1 = F.sigmoid(pred_x_tp1)
        return pred_x_tp1, message, frame_predictor_hidden#, h_t, c_t, delta_c_list, delta_m_list

    def predict(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1.data, message.data, memory #, h_t, c_t, delta_c_list, delta_m_list

class FCN8s_1_agent_4_view(nn.Module):
    def __init__(self, num_classes, pretrained=True, caffe=False):
        super(FCN8s_1_agent_4_view, self).__init__()
        vgg = models.vgg16()
        self.encoder = nn.Conv2d(3*4, 3, kernel_size=1)
        self.num_classes = num_classes * 4
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        '''
        100 padding for 2 reasons:
            1) support very small input size
            2) allow cropping in order to match size of different layers' feature maps
        Note that the cropped part corresponds to a part of the 100 padding
        Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
        '''
        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        # print('features: ', features)
        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, self.num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc7 = nn.Conv2d(512, (12*12) * self.num_classes, kernel_size=1)
        self.score_fr = nn.Sequential(
            # fc6, nn.ReLU(), nn.Dropout(),
            fc7, # nn.ReLU(), nn.Dropout(), score_fr
        )
        self.upscore2 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=4, stride=1, bias=False)

    def forward(self, x, m_t, m_t_others, m_2, m_3):
        x = x[:, 0].permute(0, 3, 1, 2)  # 1 x.shape:  torch.Size([5, 5, 128, 128])
        # print(1, 'x.shape: ', x.shape)
        x = self.encoder(x) # 1 x.shape:  torch.Size([5, 3, 128, 128])
        # print(2, 'x.shape: ', x.shape)
        x_size = x.size()
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)
        score_fr = self.score_fr(pool5)
        # print('score_fr.shape: ', score_fr.shape)  # score_fr.shape:  torch.Size([5, 84, 5, 5])  # score_fr.shape:  torch.Size([5, 84, 11, 11])
        # print('4096 // self.num_classes * 5: ', 4096 // self.num_classes * 5)
        score_fr = score_fr.reshape(score_fr.shape[0], score_fr.shape[1] // (12*12),
                                    score_fr.shape[2] * 12, score_fr.shape[3] * 12)
        upscore2 = self.upscore2(score_fr)[:, :, 0:x_size[2], 0:x_size[3]]
        upscore8 = upscore2
        return upscore8  # [:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()
'''
class FCN8s_1_agent_4_view_v0(nn.Module):
    def __init__(self, num_classes, pretrained=True, caffe=False):
        super(FCN8s_1_agent_4_view, self).__init__()
        vgg = models.vgg16()
        self.encoder = nn.Conv2d(3*4, 3, kernel_size=1)
        num_classes = num_classes * 4
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        # 100 padding for 2 reasons:
        #     1) support very small input size
        #     2) allow cropping in order to match size of different layers' feature maps
        # Note that the cropped part corresponds to a part of the 100 padding

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        # self.decoder = nn.Conv2d(num_classes, num_classes*2, kernel_size=1)
        # self.upscore2.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        # self.upscore_pool4.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        # self.upscore8.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 16))

    def forward(self, x, m_t, m_t_others, m_2, m_3):
        # print('x.shape, m_t.shape, m_t_others.shape, m_2.shape, m_1.shape: ', x.shape, m_t.shape, m_t_others.shape, m_2.shape, m_3.shape)
        # x.shape, m_t.shape, m_t_others.shape, m_2.shape, m_1.shape:  torch.Size([5, 1, 128, 128, 12]) torch.Size([5, 1, 128, 128, 1]) torch.Size([5, 1, 128, 128, 1]) torch.Size([5, 1, 128, 128, 1]) torch.Size([5, 1, 128, 128, 1])
        # x = torch.cat([x, m_t, m_t_others, m_2, m_3], -1)
        x = x[:, 0].permute(0, 3, 1, 2)  # 1 x.shape:  torch.Size([5, 5, 128, 128])
        # print(1, 'x.shape: ', x.shape)
        x = self.encoder(x) # 1 x.shape:  torch.Size([5, 3, 128, 128])
        # print(2, 'x.shape: ', x.shape)
        x_size = x.size()
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                           + upscore2)

        score_pool3 = self.score_pool3(0.0001 * pool3)
        # print('score_pool3.shape: ', score_pool3.shape)  # score_pool3.shape:  torch.Size([5, 21, 41, 41])
        upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])]
                                 + upscore_pool4)
        # print('upscore8: ', upscore8.shape)  # upscore8:  torch.Size([5, 21, 216, 216])
        # upscore8 = self.decoder(upscore8)
        return upscore8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()

class CONV_NN_v3(nn.Module):
    # frame prediction
    def __init__(self, input_dim, h_units, act, args):
        super(CONV_NN_v3, self).__init__()
        # [10, 128, 128, 5]
        self.filter_size = args.filter_size
        self.padding = self.filter_size // 2

        self.frame_predictor = DeterministicConvLSTM(input_dim, h_units[-1], h_units[0], len(h_units), args)
        self.l3 = nn.Conv3d(h_units[-1], 3, kernel_size=self.filter_size, stride=1, padding=self.padding, bias=False)

        if act == "relu":
            self.act = F.relu
        elif act == "sig":
            self.act = F.sigmoid

    def __call__(self, x, memory):
        pred_x_tp1, memory = self.forward(x, memory)
        return pred_x_tp1, memory #, h_t, c_t, delta_c_list, delta_m_list

    def forward(self, x_t, frame_predictor_hidden):
        # print('x_t: {}, m_t: {}, m_t_others: {}'.format(x_t.shape, m_t.shape, m_t_others.shape))  # [10, 3, 128, 128, 3], [10, 1, 128, 128, 1], [10, 1, 128, 128, 1]
        # m_t = m_t.repeat(1, x_t.shape[1] // m_t.shape[1], 1, 1, 1)
        # m_t_others = m_t_others.repeat(1, x_t.shape[1] // m_t_others.shape[1], 1, 1, 1)
        x = torch.cat([x_t], -1)
        x = x.permute(0, 4, 1, 2, 3)
        h, frame_predictor_hidden = self.frame_predictor(x, frame_predictor_hidden)
        pred_x_tp1 = self.l3(h)
        message = None
        pred_x_tp1 = pred_x_tp1.permute(0, 2, 3, 4, 1)
        pred_x_tp1 = F.sigmoid(pred_x_tp1)
        return pred_x_tp1 , frame_predictor_hidden#, h_t, c_t, delta_c_list, delta_m_list

    def predict(self, x, memory):
        pred_x_tp1, memory = self.forward(x, memory)
        return pred_x_tp1.data, memory #, h_t, c_t, delta_c_list, delta_m_list

class CONV_NN_v3_2_agent(nn.Module):
    # frame prediction
    def __init__(self, input_dim, h_units, act, args):
        super(CONV_NN_v3_2_agent, self).__init__()
        # [10, 128, 128, 5]
        self.filter_size = args.filter_size
        self.padding = self.filter_size // 2

        self.frame_predictor = DeterministicConvLSTM(input_dim, h_units[-1], h_units[0], len(h_units), args)
        self.l3 = nn.Conv3d(h_units[-1], 3, kernel_size=self.filter_size, stride=1, padding=self.padding, bias=False)

        if act == "relu":
            self.act = F.relu
        elif act == "sig":
            self.act = F.sigmoid

    def __call__(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1, message, memory #, h_t, c_t, delta_c_list, delta_m_list

    def forward(self, x_t, m_t, m_t_others, frame_predictor_hidden):
        # print('x_t: {}, m_t: {}, m_t_others: {}'.format(x_t.shape, m_t.shape, m_t_others.shape))  # [10, 3, 128, 128, 3], [10, 1, 128, 128, 1], [10, 1, 128, 128, 1]
        # m_t = m_t.repeat(1, x_t.shape[1] // m_t.shape[1], 1, 1, 1)
        # m_t_others = m_t_others.repeat(1, x_t.shape[1] // m_t_others.shape[1], 1, 1, 1)
        x = torch.cat([x_t, m_t, m_t_others, ], -1)
        x = x.permute(0, 4, 1, 2, 3)
        h, frame_predictor_hidden = self.frame_predictor(x, frame_predictor_hidden)
        pred_x_tp1 = self.l3(h)
        message = None
        pred_x_tp1 = pred_x_tp1.permute(0, 2, 3, 4, 1)
        pred_x_tp1 = F.sigmoid(pred_x_tp1)
        return pred_x_tp1, message, frame_predictor_hidden#, h_t, c_t, delta_c_list, delta_m_list

    def predict(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1.data, message.data, memory #, h_t, c_t, delta_c_list, delta_m_list

class FCN8s_1_agent(nn.Module):
    def __init__(self, num_classes, pretrained=True, caffe=False):
        super(FCN8s_1_agent, self).__init__()
        vgg = models.vgg16()
        self.encoder = nn.Conv2d(6, 3, kernel_size=1)
        num_classes = num_classes * 2
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        # 100 padding for 2 reasons:
        #     1) support very small input size
        #     2) allow cropping in order to match size of different layers' feature maps
        # Note that the cropped part corresponds to a part of the 100 padding
        # Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
        
        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        # self.decoder = nn.Conv2d(num_classes, num_classes*2, kernel_size=1)
        # self.upscore2.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        # self.upscore_pool4.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        # self.upscore8.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 16))

    def forward(self, x, m_t, m_t_others):
        # x = torch.cat([x, m_t, m_t_others], -1)
        x = x[:, 0].permute(0, 3, 1, 2)  # 1 x.shape:  torch.Size([5, 5, 128, 128])
        # print(1, 'x.shape: ', x.shape)
        x = self.encoder(x) # 1 x.shape:  torch.Size([5, 3, 128, 128])
        # print(2, 'x.shape: ', x.shape)
        x_size = x.size()
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                           + upscore2)

        score_pool3 = self.score_pool3(0.0001 * pool3)
        # print('score_pool3.shape: ', score_pool3.shape)  # score_pool3.shape:  torch.Size([5, 21, 41, 41])
        upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])]
                                 + upscore_pool4)
        # print('upscore8: ', upscore8.shape)  # upscore8:  torch.Size([5, 21, 216, 216])
        # upscore8 = self.decoder(upscore8)
        return upscore8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()


class FCN8s_2_agent(nn.Module):
    def __init__(self, num_classes, pretrained=True, caffe=False):
        super(FCN8s_2_agent, self).__init__()
        vgg = models.vgg16()
        # if pretrained:
        #     if caffe:
        #         # load the pretrained vgg16 used by the paper's author
        #         vgg.load_state_dict(torch.load(vgg16_caffe_path))
        #     else:
        #         vgg.load_state_dict(torch.load(vgg16_path))
        self.encoder = nn.Conv2d(5, 3, kernel_size=1)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        
        # 100 padding for 2 reasons:
        #     1) support very small input size
        #     2) allow cropping in order to match size of different layers' feature maps
        # Note that the cropped part corresponds to a part of the 100 padding
        # Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3.weight.data.zero_()
        self.score_pool3.bias.data.zero_()
        self.score_pool4.weight.data.zero_()
        self.score_pool4.bias.data.zero_()

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        # self.upscore2.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        # self.upscore_pool4.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        # self.upscore8.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 16))

    def forward(self, x, m_t, m_t_others):
        x = torch.cat([x, m_t, m_t_others], -1)
        x = x[:, 0].permute(0, 3, 1, 2)  # 1 x.shape:  torch.Size([5, 5, 128, 128])
        # print(1, 'x.shape: ', x.shape)
        x = self.encoder(x) # 1 x.shape:  torch.Size([5, 3, 128, 128])
        # print(2, 'x.shape: ', x.shape)
        x_size = x.size()
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)

        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)

        score_pool4 = self.score_pool4(0.01 * pool4)
        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                           + upscore2)

        score_pool3 = self.score_pool3(0.0001 * pool3)
        upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])]
                                 + upscore_pool4)
        return upscore8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()

class CONV_NN_v3_1_agent(nn.Module):
    # frame prediction
    def __init__(self, input_dim, h_units, act, args):
        super(CONV_NN_v3_1_agent, self).__init__()
        # [10, 128, 128, 5]
        self.filter_size = args.filter_size
        self.padding = self.filter_size // 2

        self.frame_predictor = DeterministicConvLSTM(input_dim, h_units[-1], h_units[0], len(h_units), args)
        self.l3 = nn.Conv3d(h_units[-1], 6, kernel_size=self.filter_size, stride=1, padding=self.padding, bias=False)

        if act == "relu":
            self.act = F.relu
        elif act == "sig":
            self.act = F.sigmoid

    def __call__(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1, message, memory #, h_t, c_t, delta_c_list, delta_m_list

    def forward(self, x_t, m_t, m_t_others, frame_predictor_hidden):
        # print('x_t: {}, m_t: {}, m_t_others: {}'.format(x_t.shape, m_t.shape, m_t_others.shape))  # [10, 3, 128, 128, 3], [10, 1, 128, 128, 1], [10, 1, 128, 128, 1]
        # m_t = m_t.repeat(1, x_t.shape[1] // m_t.shape[1], 1, 1, 1)
        # m_t_others = m_t_others.repeat(1, x_t.shape[1] // m_t_others.shape[1], 1, 1, 1)
        # x = torch.cat([x_t, m_t, m_t_others, ], -1)
        x = torch.cat([x_t, ], -1)
        x = x.permute(0, 4, 1, 2, 3)
        h, frame_predictor_hidden = self.frame_predictor(x, frame_predictor_hidden)
        pred_x_tp1 = self.l3(h)
        message = None
        pred_x_tp1 = pred_x_tp1.permute(0, 2, 3, 4, 1)
        pred_x_tp1 = F.sigmoid(pred_x_tp1)
        return pred_x_tp1, message, frame_predictor_hidden#, h_t, c_t, delta_c_list, delta_m_list

    def predict(self, x, m_t, m_t_others, memory):
        pred_x_tp1, message, memory = self.forward(x, m_t, m_t_others, memory)
        return pred_x_tp1.data, message.data, memory #, h_t, c_t, delta_c_list, delta_m_list
'''

def process_img(listener_i, image, sensor_queue):
    image.convert(cc.Raw)

    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    # array = array.swapaxes(0, 1)
    # print('image event: ')
    image.save_to_disk('_out_{}/{:08d}'.format(listener_i, image.frame_number))
    sensor_queue.put((image.frame, listener_i, array))

def collision_callback(listener_i, event, sensor_queue):
    print('collision event: ', event)
    sensor_queue.put(('collision_sensor', listener_i, event))

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=0,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '-tm_p', '--tm_port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=0,
        type=int)
    argparser.add_argument(
        '--gamma_correction',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--number_of_rgb_camera',
        default=4,
        type=int,
        help='number of RGB camera (default: 4)')
    argparser.add_argument(
        '--camera_height',
        default=15.,
        type=float,
        help='height of RGB camera (default: 15., 13., 60)')
    argparser.add_argument(
        '--save_time_steps',
        default=600,
        type=int,
        help='number saved time steps (default: 5000)')
    argparser.add_argument(
        '--vehicle_color',
        default='fixed',
        type=str,
        help='vehicle color: random / fixed')
    argparser.add_argument(
        '--planner_type',
        default='ImageBased',
        type=str,
        help='planner type')
    argparser.add_argument(
        '--gen_waypoints',
        default=0,
        type=int,
        help='generate waypoints')
    argparser.add_argument(
        '--num_view',
        default=4,  # 4,
        type=int,
        help='number of views')
    argparser.add_argument(
        '--planner_input_modality',
        default='t2nod',  # 'frame', # 't2nod',
        type=str,
        help='frame / polygon / t2nod')
    argparser.add_argument(
        '--baseline_type',
        default='monolith',
        type=str,
        help='allcomm / monolith / nocomm / no_replan, monolith is not in use')
    argparser.add_argument(
        '--delay_time',
        default=None,
        type=str,
        help='None / sim')
    argparser.add_argument(
        '--global_percentage_speed_difference',
        help='global_percentage_speed_difference, -160.0',
        default=None,
        type=float)
    argparser.add_argument(
        '--num_classes',
        default=10,
        type=int,
        help='num classes for T2NO/D')

    args = argparser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    sensor_list = []

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.load_world('Town02')
    client.set_timeout(10.0)
    prev_v_x = 0
    prev_v_y = 0
    prev_v_z = 0
    prev_a_x = 0
    prev_a_y = 0
    prev_a_z = 0
    try:
        traffic_manager = client.get_trafficmanager(args.tm_port)
        # vehicle.set_autopilot(True) requires traffic manager with sync mode
        traffic_manager.set_synchronous_mode(True)
        world = client.get_world()

        synchronous_master = False

        if args.sync:
            settings = world.get_settings()
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.2  # 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        # weather = carla.WeatherParameters(
        #     cloudyness=80.0,
        #     precipitation=30.0,
        #     sun_altitude_angle=70.0)
        # ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, MidRainyNoon,
        # HardRainNoon, SoftRainNoon,
        # ClearSunset, CloudySunset, WetSunset, WetCloudySunset,
        # MidRainSunset, HardRainSunset, SoftRainSunset
        # weather = carla.WeatherParameters.ClearNoon
        # weather = carla.WeatherParameters.CloudyNoon
        # weather = carla.WeatherParameters.SoftRainSunset
        # weather = carla.WeatherParameters.HardRainSunset
        # weather = carla.WeatherParameters.MidRainSunset
        # weather = carla.WeatherParameters.WetCloudySunset
        # weather = carla.WeatherParameters.WetSunset
        # weather = carla.WeatherParameters.CloudySunset
        # weather = carla.WeatherParameters.ClearSunset
        # weather = carla.WeatherParameters.SoftRainNoon
        # weather = carla.WeatherParameters.HardRainNoon
        # weather = carla.WeatherParameters.MidRainyNoon
        # weather = carla.WeatherParameters.WetCloudyNoon
        # world.set_weather(weather)
        # print(world.get_weather())

        # global sensors_callback
        # sensors_callback = []
        blueprint_library = world.get_blueprint_library()
        all_spectators = world.get_actors().filter('traffic.traffic_light*')
        print('len(spectators): {}'.format(len(all_spectators)))
        spectators = []
        for each_spectator in all_spectators:
            print(each_spectator, each_spectator.id, each_spectator.get_transform())
            # Town02, run ./CarlaUE.sh first every time
            # if each_spectator.id in [87, 88, 91, 92]:   # [, 313, 314, 316, 317]:
            if each_spectator.id in [87, 88, 91, 92, 94, 105, 106, 108]:   # [, 313, 314, 316, 317]:
                spectators.append(each_spectator)
        assert args.number_of_rgb_camera < len(all_spectators), \
            'args.number_of_rgb_camera: {} > len(spectators): {}'.format(args.number_of_rgb_camera, len(all_spectators))
        # setup sensors
        # We create the sensor queue in which we keep track of the information
        # already received. This structure is thread safe and can be
        # accessed by all the sensors callback concurrently without problem.
        sensor_queue = Queue()
        bp = blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(600))
        bp.set_attribute('image_size_y', str(600))
        bp.set_attribute('sensor_tick', str(0.0333))
        collision_bp = blueprint_library.find('sensor.other.collision')
        # Sensor00
        spectator_transform = spectators[2].get_transform()
        spectator_transform_00 = spectators[2].get_transform()

        # create sensor
        sensor_transform = carla.Transform(carla.Location(x=spectator_transform.location.x,
                                                          y=spectator_transform.location.y,
                                                          z=args.camera_height),
                                           carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
        sensor_00 = world.spawn_actor(bp, sensor_transform)

        # add callbacks
        sensor_00.listen(lambda data: process_img(listener_i=0, image=data, sensor_queue=sensor_queue))

        sensor_list.append(sensor_00)
        if args.num_view > 1:
            # Sensor01
            spectator_transform = spectators[1].get_transform()

            # create sensor
            sensor_transform = carla.Transform(carla.Location(x=spectator_transform_00.location.x,
                                                              y=spectator_transform.location.y,
                                                              z=args.camera_height),
                                               carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
            sensor_01 = world.spawn_actor(bp, sensor_transform)

            # add callbacks
            sensor_01.listen(lambda data: process_img(listener_i=1, image=data, sensor_queue=sensor_queue))

            sensor_list.append(sensor_01)
            if args.num_view > 2:
                # Sensor02
                # spectator_transform = spectators[3].get_transform()
                # spectator_transform_02 = spectators[3].get_transform()
                spectator_transform = spectators[0].get_transform()
                spectator_transform_02 = spectators[0].get_transform()
                print('spectator_transform.location: ', spectator_transform.location)

                # create sensor
                sensor_transform = carla.Transform(carla.Location(x=spectator_transform.location.x + 40, # "+" means the camera is moving up
                                                                  y=spectator_transform.location.y,
                                                                  z=args.camera_height),
                                                   carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
                sensor_02 = world.spawn_actor(bp, sensor_transform)

                # add callbacks
                sensor_02.listen(lambda data: process_img(listener_i=2, image=data, sensor_queue=sensor_queue))

                sensor_list.append(sensor_02)
                # Sensor03
                # spectator_transform = spectators[0].get_transform()
                spectator_transform = spectators[3].get_transform()

                # create sensor
                sensor_transform = carla.Transform(carla.Location(x=spectator_transform_02.location.x + 40,
                                                                  y=spectator_transform.location.y,
                                                                  z=args.camera_height),
                                                   carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
                sensor_03 = world.spawn_actor(bp, sensor_transform)

                # add callbacks
                sensor_03.listen(lambda data: process_img(listener_i=3, image=data, sensor_queue=sensor_queue))

                sensor_list.append(sensor_03)
                if args.num_view > 4:
                    # Sensor04
                    spectator_transform = spectators[4].get_transform()
                    spectator_transform_04 = spectators[4].get_transform()

                    # create sensor
                    sensor_transform = carla.Transform(carla.Location(x=spectator_transform.location.x,
                                                                      y=spectator_transform.location.y,
                                                                      z=args.camera_height),
                                                       carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
                    sensor_04 = world.spawn_actor(bp, sensor_transform)

                    # add callbacks
                    sensor_04.listen(lambda data: process_img(listener_i=4, image=data, sensor_queue=sensor_queue))

                    sensor_list.append(sensor_04)
                    # Sensor05
                    spectator_transform = spectators[5].get_transform()

                    # create sensor
                    sensor_transform = carla.Transform(carla.Location(x=spectator_transform_04.location.x,
                                                                      y=spectator_transform.location.y,
                                                                      z=args.camera_height),
                                                       carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
                    sensor_05 = world.spawn_actor(bp, sensor_transform)

                    # add callbacks
                    sensor_05.listen(lambda data: process_img(listener_i=5, image=data, sensor_queue=sensor_queue))

                    sensor_list.append(sensor_05)
                    # Sensor06
                    spectator_transform = spectators[6].get_transform()
                    spectator_transform_06 = spectators[6].get_transform()

                    # create sensor
                    sensor_transform = carla.Transform(carla.Location(x=spectator_transform.location.x,
                                                                      y=spectator_transform.location.y,
                                                                      z=args.camera_height),
                                                       carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
                    sensor_06 = world.spawn_actor(bp, sensor_transform)

                    # add callbacks
                    sensor_06.listen(lambda data: process_img(listener_i=6, image=data, sensor_queue=sensor_queue))

                    sensor_list.append(sensor_06)
                    # Sensor07
                    spectator_transform = spectators[7].get_transform()

                    # create sensor
                    sensor_transform = carla.Transform(carla.Location(x=spectator_transform_06.location.x,
                                                                      y=spectator_transform.location.y,
                                                                      z=args.camera_height),
                                                       carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))
                    sensor_07 = world.spawn_actor(bp, sensor_transform)

                    # add callbacks
                    sensor_07.listen(lambda data: process_img(listener_i=7, image=data, sensor_queue=sensor_queue))

                    sensor_list.append(sensor_07)

        # add a world.tick after seeing up the cameras. So that they are created before all the vehicles.
        # That way you more guarantee these actors are spawned in the world before trying to spawn all the other
        # actors likes cars and walkers. It's possible the cameras are being spawned in between other actors and
        # therefore starting there frame grabs at different times.
        if not args.sync or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        blueprints = world.get_blueprint_library().filter(args.filterv)
        # blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        first_agent = 220 + args.num_view
        second_agent = 255 + args.num_view
        for n, transform in enumerate(spawn_points):
            print('n: ',  n)
            if n >= args.number_of_vehicles:
                break
            # blueprint = random.choice(blueprints)
            blueprint = blueprints[0]
            # print('blueprint.id: ', blueprint.id)  #  vehicle.audi.a2
            if blueprint.has_attribute('color'):
                if args.vehicle_color == 'random':
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                else:
                    color = blueprint.get_attribute('color').recommended_values[0]
                blueprint.set_attribute('color', color)
                #  ['224,0,0', '134,134,134', '66,66,66', '108,109,126', '255,255,255', '0,0,0']
                # print('color: ', color, blueprint.get_attribute('color').recommended_values)
                if n == (10 + args.num_view):  # +210:
                    color = blueprint.get_attribute('color').recommended_values[1]
                    print(n, color)
                    blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                # driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                driver_id = blueprint.get_attribute('driver_id').recommended_values[0]
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            # print('blueprint.id: ', blueprint.id)
            vehicle = world.spawn_actor(blueprint, transform)
            vehicles_list.append(vehicle)
            # print('created {}'.format(vehicle.id))

            # Let's put the vehicle to drive around.
            # if vehicle.id not in [228, 256]:  # 261]:  # , 261]:
            if vehicle.id not in [first_agent]:  # , second_agent]:
                vehicle.set_autopilot(True)
                print('autopilot {}'.format(vehicle.id))
            # batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
        print('synchronous_master: ', synchronous_master)
        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not args.sync or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters, speed
        if args.global_percentage_speed_difference is not None:
            traffic_manager.global_percentage_speed_difference(args.global_percentage_speed_difference)
        img_counter = 0

        vehicles = world.get_actors().filter('vehicle.*')
        for i, each_vehicle in enumerate(vehicles):
            if each_vehicle.id in [first_agent]:  # , 261
                vehicle_id_228 = each_vehicle
                print(i, first_agent)
                collision_sensor_00 = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle_id_228)
                # collision_sensor_00 = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
                collision_sensor_00.listen(
                    lambda data: collision_callback(listener_i=0, event=data, sensor_queue=sensor_queue))
                sensor_list.append(collision_sensor_00)
                # print(sensor_list, sensor_list[0], sensor_list[1])
            elif each_vehicle.id in [second_agent]:  # , 261
                vehicle_id_261 = each_vehicle
                print(i, second_agent)

        agent = BasicAgent(vehicle_id_228, planner_type=args.planner_type, planner_input_modality=args.planner_input_modality, args=args)
        spawn_point = world.get_map().get_spawn_points()[0]
        agent.set_destination((spawn_point.location.x,
                               spawn_point.location.y,
                               spawn_point.location.z))

        # agent2 = BasicAgent(vehicle_id_261, planner_type=args.planner_type)
        # spawn_point = world.get_map().get_spawn_points()[0]
        # agent2.set_destination((spawn_point.location.x,
        #                        spawn_point.location.y,
        #                        spawn_point.location.z))

        start_time = time.time()
        velocity_list = []
        prev_loc_x = 0
        prev_loc_y = 0
        while img_counter < args.save_time_steps:
            # print('args.sync:{} and synchronous_master: {}'.format(args.sync, synchronous_master))
            # args.sync:True and synchronous_master: True
            if args.sync and synchronous_master:
                world.tick()
                w_frame = world.get_snapshot().frame
                print('\nWorld\'s frame: {}'.format(w_frame))
                view_img_dict = {}
                collision_info_dict = {0: []}
                try:
                    for i in range(len(sensor_list)):
                        s_frame = sensor_queue.get(True, 1.0)   # block=True, timeout=None
                        # print(i, s_frame)
                        if s_frame[0] == 'collision_sensor':
                            print(' Collision: {}, Sensor: {}, Event: {}'.format(s_frame[0], s_frame[1], s_frame[2]))
                            collision_info_dict[s_frame[1]].append(s_frame[2])
                        else:
                            print(' Frame: {} Sensor: {}'.format(s_frame[0], s_frame[1]))
                            view_img_dict[s_frame[1]] = s_frame[2]
                except Empty:
                    print(' Some of the collision/RGB sensor infomation is missed')
                img_counter = img_counter + 1

                # plt.imshow(view_img_dict[0])
                # plt.show()
                v = agent._vehicle.get_velocity()
                loc = agent._vehicle.get_location()
                print('v_x: {}, v_y: {}, v_z: {}'.format(v.x, v.y, v.z))
                print('loc_x: {}, loc_y: {}, loc_z: {}'.format(loc.x, loc.y, loc.z))
                a = agent._vehicle.get_acceleration()
                # print('a_x: {}, a_y: {}, a_z: {}'.format(a.x, a.y, a.z))
                if len(view_img_dict) == args.num_view:
                    control = agent.run_step(debug=True, view_img=view_img_dict, img_step=img_counter)
                if not agent.metrics['GoalReached']:
                    agent.metrics['TotalControlEffort'] += (a.x ** 2 + a.y ** 2) ** 0.5
                    if img_counter > 1:
                        agent.metrics['TravelDistance'] += 8 * ((loc.x - prev_loc_x) ** 2 + (loc.y - prev_loc_y) ** 2) ** 0.5
                    prev_loc_x = loc.x
                    prev_loc_y = loc.y
                    if len(agent.metrics['TimeToReachGoals']) == 0:
                        agent.metrics['TimeToReachGoals'].append(time.time() - start_time)
                    else:
                        agent.metrics['TimeToReachGoals'][-1] = time.time() - start_time
                    if len(collision_info_dict) > 0:
                        agent.metrics['NumberOfCollision'] += len(collision_info_dict[0])
                    if prev_a_x * a.x < 0 or prev_a_y * a.y < 0: # or prev_v_z * v.z < 0:
                        agent.metrics['NumberOfSuddenReversalsInOrder'] += 1
                if args.num_view == 4:
                    control.throttle *= 0.6
                prev_v_x = v.x
                prev_v_y = v.y
                prev_v_z = v.z
                prev_a_x = a.x
                prev_a_y = a.y
                prev_a_z = a.z
                velocity_list.append([prev_v_x, prev_v_y, prev_v_z])
                control.manual_gear_shift = False
                print('1 control: ', control)
                print('Metrics: ', agent.metrics)
                # control.throttle = 0.0
                # control.brake = 1.0
                # print('2 control: ', control)
                vehicle_id_228.apply_control(control)

                # control = agent2.run_step(debug=True)
                # control.manual_gear_shift = False
                # print('2 control: ', control)
                # vehicle_id_261.apply_control(control)
                '''
                if args.gen_waypoints:
                    d = 2.0
                    waypoint01 = world.get_map().get_waypoint(vehicle_id_228.get_location(), project_to_road=True,
                                                              lane_type=(carla.LaneType.Driving))  # | carla.LaneType.Sidewalk
                    waypoint02_list = []
                    for i in range(10):
                        waypoint02 = waypoint01.next(d)  # [0]
                        waypoint02_list.extend(waypoint02)
                        waypoint01 = waypoint02[-1]

                    for w in waypoint02_list:
                        t = w.transform
                        begin = t.location + carla.Location(z=1.0)
                        angle = math.radians(t.rotation.yaw)
                        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
                        color = carla.Color()
                        color.r, color.g, color.b, color.a = 255, 0, 0, 1  # r, g, b
                        world.debug.draw_arrow(begin, end, arrow_size=0.1, life_time=1.0,
                                                    color=color)

                    waypoint01 = world.get_map().get_waypoint(vehicle_id_261.get_location(), project_to_road=True,
                                                              lane_type=(carla.LaneType.Driving))  # | carla.LaneType.Sidewalk
                    waypoint02_list = []
                    for i in range(10):
                        waypoint02 = waypoint01.next(d)  # [0]
                        waypoint02_list.extend(waypoint02)
                        waypoint01 = waypoint02[-1]

                    for w in waypoint02_list:
                        t = w.transform
                        begin = t.location + carla.Location(z=1.0)
                        angle = math.radians(t.rotation.yaw)
                        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
                        color = carla.Color()
                        color.r, color.g, color.b, color.a = 0, 255, 0, 1  # r, g, b
                        world.debug.draw_arrow(begin, end, arrow_size=0.1, life_time=1.0,
                                                    color=color)
                '''
            else:
                world.wait_for_tick()

        np.save('velocity_list.npy', velocity_list)

    finally:
        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d sensors' % len(sensor_list))
        for sensor in sensor_list:
            sensor.stop()
            sensor.destroy()

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
