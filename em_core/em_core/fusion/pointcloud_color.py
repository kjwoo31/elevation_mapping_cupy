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
from .fusion_utils import add_color_kernel
from .fusion_utils import color_average_kernel


class Color(FusionBase):

    def __init__(self, params, *args, **kwargs):
        self.name = 'pointcloud_color'
        self.cell_n = params.cell_n
        self.resolution = params.resolution

        self.add_color_kernel = add_color_kernel(params.cell_n, params.cell_n,)
        self.color_average_kernel = color_average_kernel(self.cell_n, self.cell_n)

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
        self.color_map = cp.zeros(
            (1 + 3 * layer_ids.shape[0], self.cell_n, self.cell_n),
            dtype=cp.uint32)
        points_all = points_all.astype(cp.float32)
        self.add_color_kernel(
            points_all,
            rotation,
            translation,
            pcl_ids,
            layer_ids,
            cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
            self.color_map,
            size=(points_all.shape[0]))
        self.color_average_kernel(
            self.color_map,
            pcl_ids,
            layer_ids,
            cp.array([points_all.shape[1], pcl_ids.shape[0]], dtype=cp.int32),
            semantic_map,
            size=(self.cell_n * self.cell_n))
