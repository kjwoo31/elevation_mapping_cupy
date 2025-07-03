#!/usr/bin/env python3
# Copyright 2024 ROBOTIS CO., LTD.
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
#
# author : Jinwoo Kim, original author: Takahiro Miki

import cupy as cp

import torch
import torch.nn as nn


class TraversabilityFilter(nn.Module):

    def __init__(self, w1, w2, w3, w_out, device='cuda', use_bias=False):
        super(TraversabilityFilter, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, dilation=1, padding=0, bias=use_bias)
        self.conv2 = nn.Conv2d(1, 4, 3, dilation=2, padding=0, bias=use_bias)
        self.conv3 = nn.Conv2d(1, 4, 3, dilation=3, padding=0, bias=use_bias)
        self.conv_out = nn.Conv2d(12, 1, 1, bias=use_bias)

        self.conv1.weight = nn.Parameter(torch.from_numpy(w1).float())
        self.conv2.weight = nn.Parameter(torch.from_numpy(w2).float())
        self.conv3.weight = nn.Parameter(torch.from_numpy(w3).float())
        self.conv_out.weight = nn.Parameter(torch.from_numpy(w_out).float())

    def __call__(self, elevation_cupy):
        elevation_cupy = elevation_cupy.astype(cp.float32)
        elevation = torch.as_tensor(elevation_cupy, device=self.conv1.weight.device)

        with torch.no_grad():
            out1 = self.conv1(elevation.view(-1, 1, elevation.shape[0], elevation.shape[1]))
            out2 = self.conv2(elevation.view(-1, 1, elevation.shape[0], elevation.shape[1]))
            out3 = self.conv3(elevation.view(-1, 1, elevation.shape[0], elevation.shape[1]))

            out1 = out1[:, :, 2:-2, 2:-2]
            out2 = out2[:, :, 1:-1, 1:-1]
            out = torch.cat((out1, out2, out3), dim=1)
            out = self.conv_out(out.abs())
            out = torch.exp(-out)
            out_cupy = cp.asarray(out)

        return out_cupy
