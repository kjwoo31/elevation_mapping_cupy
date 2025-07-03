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

from .fusion_base import FusionBase
from .fusion_utils import exponential_correspondences_to_map_kernel


class ImageExponential(FusionBase):

    def __init__(self, params, *args, **kwargs):
        self.name = 'image_exponential'
        self.cell_n = params.cell_n
        self.resolution = params.resolution

        self.exponential_correspondences_to_map_kernel = exponential_correspondences_to_map_kernel(
            resolution=self.resolution,
            width=self.cell_n,
            height=self.cell_n,
            alpha=0.7)

    def __call__(
            self,
            sem_map_idx,
            image,
            class_idx,
            uv_correspondence,
            valid_correspondence,
            image_height,
            image_width,
            semantic_map,
            new_map):
        self.exponential_correspondences_to_map_kernel(
            semantic_map,
            sem_map_idx,
            image[class_idx],
            uv_correspondence,
            valid_correspondence,
            image_height,
            image_width,
            new_map,
            size=int(self.cell_n * self.cell_n))
        semantic_map[sem_map_idx] = new_map[sem_map_idx]
