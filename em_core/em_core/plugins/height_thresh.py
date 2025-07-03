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

import string
from typing import List

import cupy as cp

from .plugin_manager import PluginBase


class HeightThresh(PluginBase):
    """로봇 프레임 기준으로 일정 높이 이상의 장애물을 표시하는 플러그인 클래스."""

    def __init__(
            self,
            cell_n: int = 100,
            resolution: float = 0.05,
            threshold: float = 0.02,
            use_threshold: bool = 0, **kwargs):
        super().__init__()
        self.width = cell_n
        self.height = cell_n
        self.min_filtered = cp.zeros((self.width, self.height), dtype=cp.float32)

        self.base_elevation_kernel = cp.ElementwiseKernel(
            in_params='raw U map, raw U mask, raw U rotation',
            out_params='raw U newmap',
            preamble=string.Template(
                """
                __device__ int get_map_idx (int idx, int layer_n) {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }
                __device__ float get_map_x (int idx) {
                    float idx_x = idx / ${width} * ${resolution};
                    return idx_x;
                }
                __device__ float get_map_y (int idx) {
                    float idx_y = idx % ${width} * ${resolution};
                    return idx_y;
                }
                __device__ float transform_p (float x, float y, float z,
                        float r0, float r1, float r2) {
                    return r0 * x + r1 * y + r2 * z ;
                }
                """
            ).substitute(width=self.width, height=self.height, resolution=resolution),
            operation=string.Template(
                """
                U valid = mask[get_map_idx(i, 0)];
                if (valid) {
                    U rx = get_map_x(get_map_idx(i, 0));
                    U ry = get_map_y(get_map_idx(i, 0));
                    U rz = map[get_map_idx(i, 0)];
                    U z_b = transform_p(rx, ry, rz, rotation[6], rotation[7], rotation[8]);
                    if (${use_threshold} && z_b >= ${threshold}) {
                        newmap[get_map_idx(i, 0)] = 1.0;
                    }
                    else if (${use_threshold} && z_b < ${threshold}){
                        newmap[get_map_idx(i, 0)] = 0.0;
                    }
                    else{
                        newmap[get_map_idx(i, 0)] = z_b;
                    }
                }
                """
            ).substitute(threshold=threshold, use_threshold=int(use_threshold)),
            name='base_elevation_kernel')

    def __call__(
            self,
            elevation_map: cp.ndarray,
            layer_names: List[str],
            plugin_layers: cp.ndarray,
            plugin_layer_names: List[str],
            semantic_map: cp.ndarray,
            semantic_layer_names: List[str],
            rotation: cp.ndarray,
            *args) -> cp.ndarray:
        self.min_filtered = elevation_map[0].copy()
        self.base_elevation_kernel(
            elevation_map[0],
            elevation_map[2],
            rotation,
            self.min_filtered,
            size=(self.width * self.height))
        return self.min_filtered
