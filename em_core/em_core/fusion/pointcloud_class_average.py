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

from .fusion_base import FusionBase
from .fusion_utils import class_average_kernel
from .fusion_utils import sum_kernel


class ClassAverage(FusionBase):

    def __init__(self, params, *args, **kwargs):
        self.name = 'pointcloud_class_average'
        self.cell_n = params.cell_n
        self.resolution = params.resolution
        self.average_weight = params.average_weight

        self.sum_kernel = sum_kernel(self.resolution, self.cell_n, self.cell_n)
        self.class_average_kernel = class_average_kernel(
            self.cell_n,
            self.cell_n,
            self.average_weight)

    def __call__(
            self,
            points_all,
            rotation,
            translation,
            pcl_ids,
            layer_ids,
            elevation_map,
            semantic_map,
            new_map,
            *args):
        self.sum_kernel(
            points_all,
            rotation,
            translation,
            pcl_ids,
            layer_ids,
            cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
            semantic_map,
            new_map,
            size=(points_all.shape[0] * pcl_ids.shape[0]))
        self.class_average_kernel(
            new_map,
            pcl_ids,
            layer_ids,
            cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
            elevation_map,
            semantic_map,
            size=(self.cell_n * self.cell_n * pcl_ids.shape[0]))
