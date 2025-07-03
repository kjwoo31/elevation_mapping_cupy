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

from dataclasses import dataclass
from dataclasses import field
import pickle

import numpy as np
from simple_parsing.helpers import Serializable


@dataclass
class Parameter(Serializable):
    resolution: float = 0.04
    additional_layers: list = field(default_factory=lambda: ['color'])
    fusion_algorithms: list = field(
        default_factory=lambda: [
            'image_color',
            'image_exponential',
            'pointcloud_class_average',
            'pointcloud_color'
        ]
    )
    pointcloud_channel_fusions: dict = field(
        default_factory=lambda: {'rgb': 'color', 'default': 'class_average'})
    image_channel_fusions: dict = field(
        default_factory=lambda: {'rgb': 'color', 'default': 'exponential'})
    data_type: str = np.float32
    average_weight: float = 0.5

    map_length: float = 8.0
    sensor_noise_factor: float = 0.05
    mahalanobis_thresh: float = 2.0
    outlier_variance: float = 0.01
    drift_compensation_variance_inlier: float = 0.1
    time_variance: float = 0.01
    update_map_time_interval: float = 0.1

    max_variance: float = 1.0
    dilation_size: float = 2
    drift_compensation_alpha: float = 1.0

    traversability_inlier: float = 0.1
    wall_num_thresh: int = 100
    min_height_drift_count: int = 100

    max_ray_length: float = 2.0
    cleanup_step: float = 0.01
    cleanup_cos_thresh: float = 0.5
    min_valid_distance: float = 0.3
    max_valid_distance: float = 10.0
    max_height_range: float = 1.0

    max_drift: float = 0.10

    enable_drift_compensation: bool = True
    enable_visibility_cleanup: bool = True
    position_noise_thresh: float = 0.1
    orientation_noise_thresh: float = 0.1

    plugin_config_file: str = 'config/plugin_config.yaml'
    weight_file: str = 'config/weights.dat'

    initial_variance: float = 10.0
    w1: np.ndarray = field(default_factory=lambda: np.zeros((4, 1, 3, 3)))
    w2: np.ndarray = field(default_factory=lambda: np.zeros((4, 1, 3, 3)))
    w3: np.ndarray = field(default_factory=lambda: np.zeros((4, 1, 3, 3)))
    w_out: np.ndarray = field(default_factory=lambda: np.zeros((1, 12, 1, 1)))

    # 설정 불가능한 매개변수
    true_map_length: float = None
    cell_n: int = None
    true_cell_n: int = None

    def load_weights(self, filename):
        with open(filename, 'rb') as file:
            weights = pickle.load(file)
            self.w1 = weights['conv1.weight']
            self.w2 = weights['conv2.weight']
            self.w3 = weights['conv3.weight']
            self.w_out = weights['conv_final.weight']

    def get_names(self):
        return list(self.__annotations__.keys())

    def get_types(self):
        return [v.__name__ for v in self.__annotations__.values()]

    def set_value(self, name, value):
        setattr(self, name, value)

    def get_value(self, name):
        return getattr(self, name)

    def update(self):
        """지도 크기 및 해상도와 관련된 매개변수를 업데이트하는 멤버 함수."""
        # +2는 지도의 테두리
        self.cell_n = int(round(self.map_length / self.resolution)) + 2
        self.true_cell_n = round(self.map_length / self.resolution)
        self.true_map_length = self.true_cell_n * self.resolution
