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

from abc import ABC
from typing import List
from typing import Optional

import cupy as cp


class PluginBase(ABC):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(
            self,
            elevation_map: cp.ndarray,
            layer_names: List[str],
            plugin_layers: cp.ndarray,
            plugin_layer_names: List[str],
            semantic_map: cp.ndarray,
            semantic_layer_names: List[str],
            *args,
            **kwargs) -> cp.ndarray:
        """cupy행렬로 고도 지도 및 플러그인 layer를 가져오는 함수."""
        pass

    def get_layer_data(
            self,
            elevation_map: cp.ndarray,
            layer_names: List[str],
            plugin_layers: cp.ndarray,
            plugin_layer_names: List[str],
            semantic_map: cp.ndarray,
            semantic_layer_names: List[str],
            name: str) -> Optional[cp.ndarray]:
        """Layer 이름을 기준으로 고도, 플러그인 또는 의미론적 지도에서 layer 데이터를 불러오는 멤버 함수."""
        if name in layer_names:
            idx = layer_names.index(name)
            layer = elevation_map[idx].copy()
        elif name in plugin_layer_names:
            idx = plugin_layer_names.index(name)
            layer = plugin_layers[idx].copy()
        elif name in semantic_layer_names:
            idx = semantic_layer_names.index(name)
            layer = semantic_map[idx].copy()
        else:
            print(f'Could not find layer {name}!')
            layer = None
        return layer
