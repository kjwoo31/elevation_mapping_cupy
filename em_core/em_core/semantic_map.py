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

import re
from typing import Dict
from typing import List

import cupy as cp

from em_core.fusion.fusion_manager import FusionManager
from em_core.parameter import Parameter

import numpy as np


class SemanticMap:
    """RGB 혹은 Segmentation 이미지 정보를 반영한 의미론적 지도 관리 클래스."""

    def __init__(self, param: Parameter):

        self.param = param

        self.layer_specs_points = {}
        self.layer_specs_image = {}
        self.layer_names = []
        self.unique_fusion = []
        self.unique_data = []
        self.elements_to_shift = {}

        self.unique_fusion = self.param.fusion_algorithms

        self.amount_layer_names = len(self.layer_names)

        self.semantic_map = cp.zeros(
            (self.amount_layer_names, self.param.cell_n, self.param.cell_n),
            dtype=np.float32)
        self.new_map = cp.zeros(
            (self.amount_layer_names, self.param.cell_n, self.param.cell_n),
            np.float32)

        self.delete_new_layers = cp.ones(self.new_map.shape[0], cp.bool8)
        self.fusion_manager = FusionManager(self.param)

    def clear(self):
        self.semantic_map *= 0.0

    def initialize_fusion(self):
        for fusion in self.unique_fusion:
            if 'pointcloud_class_bayesian' == fusion:
                pcl_ids = self.get_layer_indices('class_bayesian', self.layer_specs_points)
                self.delete_new_layers[pcl_ids] = 0
            if 'pointcloud_class_max' == fusion:
                pcl_ids = self.get_layer_indices('class_max', self.layer_specs_points)
                self.delete_new_layers[pcl_ids] = 0
                layer_count = self.param.fusion_algorithms.count('class_max')
                id_max = cp.zeros(
                    (layer_count, self.param.cell_n, self.param.cell_n),
                    dtype=cp.uint32,)
                self.elements_to_shift['id_max'] = id_max
            self.fusion_manager.register_plugin(fusion)

    def update_fusion_setting(self):
        for fusion in self.unique_fusion:
            if 'pointcloud_class_bayesian' == fusion:
                pcl_ids = self.get_layer_indices('class_bayesian', self.layer_specs_points)
                self.delete_new_layers[pcl_ids] = 0
            if 'pointcloud_class_max' == fusion:
                pcl_ids = self.get_layer_indices('class_max', self.layer_specs_points)
                self.delete_new_layers[pcl_ids] = 0
                layer_count = self.param.fusion_algorithms.count('class_max')
                id_max = cp.zeros(
                    (layer_count, self.param.cell_n, self.param.cell_n),
                    dtype=cp.uint32)
                self.elements_to_shift['id_max'] = id_max

    def add_layer(self, name):
        if name not in self.layer_names:
            self.layer_names.append(name)
            self.semantic_map = cp.append(
                self.semantic_map,
                cp.zeros((1, self.param.cell_n, self.param.cell_n), dtype=np.float32),
                axis=0)
            self.new_map = cp.append(
                self.new_map,
                cp.zeros((1, self.param.cell_n, self.param.cell_n), dtype=np.float32),
                axis=0)
            self.delete_new_layers = cp.append(
                self.delete_new_layers,
                cp.array([1], dtype=cp.bool8))

    def pad_value(self, x, shift_value, idx=None, value=0.0):
        """이동량에 따라 지도에 패딩을 만드는 멤버 함수."""
        if idx is None:
            if shift_value[0] > 0:
                x[:, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[:, shift_value[0]:, :] = value
            if shift_value[1] > 0:
                x[:, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[:, :, shift_value[1]:] = value
        else:
            if shift_value[0] > 0:
                x[idx, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[idx, shift_value[0]:, :] = value
            if shift_value[1] > 0:
                x[idx, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[idx, :, shift_value[1]:] = value

    def shift_map_xy(self, shift_value):
        self.semantic_map = cp.roll(self.semantic_map, shift_value, axis=(1, 2))
        self.pad_value(self.semantic_map, shift_value, value=0.0)
        self.new_map = cp.roll(self.new_map, shift_value, axis=(1, 2))
        self.pad_value(self.new_map, shift_value, value=0.0)
        for el in self.elements_to_shift.values():
            el = cp.roll(el, shift_value, axis=(1, 2))
            self.pad_value(el, shift_value, value=0.0)

    def get_fusion(
            self,
            channels: List[str],
            channel_fusions: Dict[str, str],
            layer_specs: Dict[str, str]) -> List[str]:
        """포인트클라우드에 적용할 모든 fusion 알고리즘을 가져오는 멤버 함수."""
        fusion_list = []
        process_channels = []
        for channel in channels:
            if channel not in layer_specs:
                # 채널이 layer_specs에 없는 경우 기본 fusion 알고리즘을 사용한다.
                matched_fusion = self.get_matching_fusion(channel, channel_fusions)
                if matched_fusion is None:
                    if 'default' in channel_fusions:
                        default_fusion = channel_fusions['default']
                        print(
                            f'Layer {channel} not found in layer_specs. Using '
                            + '{default_fusion} algorithm as default.'
                        )
                        layer_specs[channel] = default_fusion
                        self.update_fusion_setting()
                    # 기본 fusion 알고리즘이 없으면 건너뛴다.
                    else:
                        print(
                            f'Layer {channel} not found in layer_specs ({layer_specs}) '
                            + 'and no default fusion is configured. Skipping.'
                        )
                        continue
                else:
                    layer_specs[channel] = matched_fusion
                    self.update_fusion_setting()
            x = layer_specs[channel]
            fusion_list.append(x)
            process_channels.append(channel)
        return process_channels, fusion_list

    def get_matching_fusion(self, channel: str, fusion_algorithms: Dict[str, str]):
        """Fusion 알고리즘이 대응하는 채널 이름과 일치하는지 확인하는 멤버 함수."""
        for fusion_algorithm, algorithm_value in fusion_algorithms.items():
            if re.match(f'^{fusion_algorithm}$', channel):
                return algorithm_value
        return None

    def get_layer_indices(self, fusion_algorithm, layer_specs):
        """특정 fusion 알고리즘에 사용되는 레이어의 인덱스를 반환하는 멤버 함수."""
        layer_indices = cp.array([], dtype=cp.int32)
        for it, (key, val) in enumerate(layer_specs.items()):
            if key in val == fusion_algorithm:
                layer_indices = cp.append(layer_indices, it).astype(cp.int32)
        return layer_indices

    def get_indices_fusion(
            self,
            pcl_channels: List[str],
            fusion_algorithm: str,
            layer_specs: Dict[str, str]):
        """Fusion 알고리즘을 적용할 포인트클라우드와 layer 인덱스를 불러오는 멤버 함수."""
        pcl_val_list = [layer_specs[x] for x in pcl_channels]
        pcl_indices = cp.array(
            [idp + 3 for idp, x in enumerate(pcl_val_list) if x == fusion_algorithm],
            dtype=cp.int32)
        layer_indices = cp.array([], dtype=cp.int32)
        for it, (key, val) in enumerate(layer_specs.items()):
            if key in pcl_channels and val == fusion_algorithm:
                layer_idx = self.layer_names.index(key)
                layer_indices = cp.append(layer_indices, layer_idx).astype(cp.int32)
        return pcl_indices, layer_indices

    def update_layers_pointcloud(self, points_all, channels, rotation, translation, elevation_map):
        """포인트클라우드를 활용하여 의미론적 지도를 업데이트하는 멤버 함수. map에서 sensor로의 rotation, translation 이용."""
        process_channels, additional_fusion = self.get_fusion(
            channels,
            self.param.pointcloud_channel_fusions,
            self.layer_specs_points)
        for channel in process_channels:
            if channel not in self.layer_names:
                print(f'Layer {channel} not found, adding it to the semantic map')
                self.add_layer(channel)

        self.new_map[self.delete_new_layers] = 0.0
        for fusion in list(set(additional_fusion)):
            pcl_ids, layer_ids = self.get_indices_fusion(
                process_channels,
                fusion,
                self.layer_specs_points)
            self.fusion_manager.execute_plugin(
                fusion,
                points_all,
                rotation,
                translation,
                pcl_ids,
                layer_ids,
                elevation_map,
                self.semantic_map,
                self.new_map,
                self.elements_to_shift)

    def update_layers_image(
            self,
            image: cp._core.core.ndarray,
            channels: List[str],
            uv_correspondence: cp._core.core.ndarray,
            valid_correspondence: cp._core.core.ndarray,
            image_height: cp._core.core.ndarray,
            image_width: cp._core.core.ndarray):
        """이미지를 활용하여 의미론적 지도를 업데이트하는 멤버 함수."""
        process_channels, fusion_methods = self.get_fusion(
            channels,
            self.param.image_channel_fusions,
            self.layer_specs_image)
        self.new_map[self.delete_new_layers] = 0.0
        for index, (fusion, channel) in enumerate(zip(fusion_methods, process_channels)):
            if channel not in self.layer_names:
                print(f'Layer {channel} not found, adding it to the semantic map')
                self.add_layer(channel)
            sem_map_idx = self.get_index(channel)

            if sem_map_idx == -1:
                print(f'Layer {channel} not found!')
                return

            self.fusion_manager.execute_image_plugin(
                fusion,
                cp.uint64(sem_map_idx),
                image,
                index,
                uv_correspondence,
                valid_correspondence,
                image_height,
                image_width,
                self.semantic_map,
                self.new_map)

    def get_map_with_name(self, name):
        # If the layer is a color layer, return the rgb map
        if name in self.layer_specs_points and self.layer_specs_points[name] == 'color':
            m = self.get_rgb(name)
            return m
        elif name in self.layer_specs_image and self.layer_specs_image[name] == 'color':
            m = self.get_rgb(name)
            return m
        else:
            m = self.get_semantic(name)
            return m

    def get_rgb(self, name):
        idx = self.layer_names.index(name)
        c = self.process_map_for_publish(self.semantic_map[idx])
        c = c.astype(np.float32)
        return c

    def get_semantic(self, name):
        idx = self.layer_names.index(name)
        c = self.process_map_for_publish(self.semantic_map[idx])
        return c

    def process_map_for_publish(self, input_map):
        """지도 패딩을 제거하는 멤버 함수."""
        m = input_map.copy()
        return m[1:-1, 1:-1]

    def get_index(self, name):
        if name not in self.layer_names:
            return -1
        else:
            return self.layer_names.index(name)
