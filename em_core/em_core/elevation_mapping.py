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

import subprocess
import threading
from typing import List

import cupy as cp

from em_core.kernels.custom_image_kernels import image_to_map_correspondence_kernel
from em_core.kernels.custom_kernels import add_points_kernel
from em_core.kernels.custom_kernels import average_map_kernel
from em_core.kernels.custom_kernels import dilation_filter_kernel
from em_core.kernels.custom_kernels import error_counting_kernel
from em_core.kernels.custom_kernels import normal_filter_kernel
from em_core.parameter import Parameter
from em_core.plugins.plugin_manager import PluginManager
from em_core.semantic_map import SemanticMap
from em_core.traversability_filter import TraversabilityFilter

import numpy as np

pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)


class ElevationMap:

    def __init__(self, param: Parameter):
        self.param = param
        self.data_type = self.param.data_type
        self.resolution = param.resolution
        self.center = cp.array([0, 0, 0], dtype=self.data_type)
        self.base_rotation = cp.eye(3, dtype=self.data_type)
        self.cell_n = param.cell_n

        self.map_lock = threading.Lock()
        self.semantic_map = SemanticMap(self.param)
        self.elevation_map = cp.zeros((7, self.cell_n, self.cell_n), dtype=self.data_type)
        self.layer_names = [
            'elevation',
            'variance',
            'is_valid',
            'traversability',
            'time',
            'upper_bound',
            'is_upper_bound']

        self.traversability_buffer = cp.full((self.cell_n, self.cell_n), cp.nan)
        self.normal_map = cp.zeros((3, self.cell_n, self.cell_n), dtype=self.data_type)

        self.initial_variance = param.initial_variance
        self.elevation_map[1] += self.initial_variance
        self.elevation_map[3] += 1.0

        self.compile_kernels()

        self.image_subscribed = False

        self.semantic_map.initialize_fusion()

        weight_file = subprocess.getoutput('echo "' + param.weight_file + '"')
        param.load_weights(weight_file)

        self.traversability_filter = TraversabilityFilter(
            param.w1,
            param.w2,
            param.w3,
            param.w_out).cuda().eval()

        self.plugin_manager = PluginManager(cell_n=self.cell_n)
        plugin_config_file = subprocess.getoutput('echo "' + param.plugin_config_file + '"')
        self.plugin_manager.load_plugin_settings(plugin_config_file)

    def clear(self):
        with self.map_lock:
            self.elevation_map *= 0.0
            self.elevation_map[1] += self.initial_variance
            self.semantic_map.clear()

    def get_map_center_position(self, position):
        position[0][:] = cp.asnumpy(self.center)

    def move(self, delta_position):
        delta_position = cp.asarray(delta_position)
        delta_pixel = cp.round(delta_position[:2] / self.resolution)
        delta_position_xy = delta_pixel * self.resolution
        self.center[:2] += cp.asarray(delta_position_xy)
        self.center[2] += cp.asarray(delta_position[2])
        self.shift_map_xy(delta_pixel)
        self.shift_map_z(-delta_position[2])

    def move_to(self, position, rotation):
        """지도를 절대 좌표로 이동하고 로봇 회전을 업데이트하는 멤버 함수."""
        self.base_rotation = cp.asarray(rotation, dtype=self.data_type)
        position = cp.asarray(position)
        delta = position - self.center
        delta_pixel = cp.around(delta[:2] / self.resolution)
        delta_xy = delta_pixel * self.resolution
        self.center[:2] += delta_xy
        self.center[2] += delta[2]
        self.shift_map_xy(-delta_pixel)
        self.shift_map_z(-delta[2])

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

    def shift_map_xy(self, delta_pixel):
        shift_value = delta_pixel.astype(cp.int32)
        if cp.abs(shift_value).sum() == 0:
            return
        with self.map_lock:
            self.elevation_map = cp.roll(self.elevation_map, shift_value, axis=(1, 2))
            self.pad_value(self.elevation_map, shift_value, value=0.0)
            self.pad_value(self.elevation_map, shift_value, idx=1, value=self.initial_variance)
            self.semantic_map.shift_map_xy(shift_value)

    def shift_map_z(self, delta_z):
        with self.map_lock:
            self.elevation_map[0] += delta_z
            self.elevation_map[5] += delta_z

    def compile_kernels(self):
        self.new_map = cp.zeros(
            (self.elevation_map.shape[0], self.cell_n, self.cell_n),
            dtype=self.data_type)
        self.traversability_input = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.traversability_mask_dummy = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.add_points_kernel = add_points_kernel(
            self.resolution,
            self.cell_n,
            self.cell_n,
            self.param.sensor_noise_factor,
            self.param.mahalanobis_thresh,
            self.param.outlier_variance,
            self.param.wall_num_thresh,
            self.param.max_ray_length,
            self.param.cleanup_step,
            self.param.min_valid_distance,
            self.param.max_height_range,
            self.param.cleanup_cos_thresh,
            self.param.enable_visibility_cleanup)
        self.error_counting_kernel = error_counting_kernel(
            self.resolution,
            self.cell_n,
            self.cell_n,
            self.param.sensor_noise_factor,
            self.param.mahalanobis_thresh,
            self.param.drift_compensation_variance_inlier,
            self.param.traversability_inlier,
            self.param.min_valid_distance,
            self.param.max_height_range)
        self.average_map_kernel = average_map_kernel(
            self.cell_n,
            self.cell_n,
            self.param.max_variance,
            self.initial_variance)

        self.dilation_filter_kernel = dilation_filter_kernel(
            self.cell_n,
            self.cell_n,
            self.param.dilation_size)
        self.normal_filter_kernel = normal_filter_kernel(self.cell_n, self.cell_n, self.resolution)

    def compile_image_kernels(self):
        """이미지 메시지 처리와 관련된 커널을 컴파일하는 멤버 함수."""
        self.valid_correspondence = cp.asarray(
            np.zeros((self.cell_n, self.cell_n), dtype=np.bool_),
            dtype=np.bool_)
        self.uv_correspondence = cp.asarray(
            np.zeros((2, self.cell_n, self.cell_n), dtype=np.float32),
            dtype=np.float32)
        self.image_to_map_correspondence_kernel = image_to_map_correspondence_kernel(
            resolution=self.resolution,
            width=self.cell_n,
            height=self.cell_n,
            tolerance_z_collision=0.10)

    def update_map_with_kernel(
            self,
            points_all,
            channels,
            rotation,
            translation,
            position_noise,
            orientation_noise):
        """고도 지도를 업데이트하는 멤버 함수."""
        self.new_map *= 0.0
        error = cp.array([0.0], dtype=cp.float32)
        error_count = cp.array([0], dtype=cp.float32)
        points = points_all[:, :3]
        with self.map_lock:
            # 고도 지도 업데이트
            translation -= self.center
            self.error_counting_kernel(
                self.elevation_map,
                points,
                cp.array([0.0], dtype=self.data_type),
                cp.array([0.0], dtype=self.data_type),
                rotation,
                translation,
                self.new_map,
                error,
                error_count,
                size=(points.shape[0]))
            if (
                self.param.enable_drift_compensation
                and error_count > self.param.min_height_drift_count
                and (
                    position_noise > self.param.position_noise_thresh
                    or orientation_noise > self.param.orientation_noise_thresh)):
                mean_error = error / error_count
                if np.abs(mean_error) < self.param.max_drift:
                    self.elevation_map[0] += mean_error * self.param.drift_compensation_alpha
            self.add_points_kernel(
                cp.array([0.0], dtype=self.data_type),
                cp.array([0.0], dtype=self.data_type),
                rotation,
                translation,
                self.normal_map,
                points,
                self.elevation_map,
                self.new_map,
                size=(points.shape[0]))
            self.average_map_kernel(
                self.new_map,
                self.elevation_map,
                size=(self.cell_n * self.cell_n))

            self.semantic_map.update_layers_pointcloud(
                points_all,
                channels,
                rotation,
                translation,
                self.new_map)

            # Traversability 업데이트
            self.traversability_input *= 0.0
            self.dilation_filter_kernel(
                self.elevation_map[5],
                self.elevation_map[2] + self.elevation_map[6],
                self.traversability_input,
                self.traversability_mask_dummy,
                size=(self.cell_n * self.cell_n))
            traversability = self.traversability_filter(self.traversability_input)
            self.elevation_map[3][3:-3, 3:-3] = traversability.reshape(
                (traversability.shape[2], traversability.shape[3]))

        # Normal vectors 업데이트
        self.update_normal(self.traversability_input)

    def update_variance(self):
        self.elevation_map[1] += self.param.time_variance * self.elevation_map[2]

    def update_time(self):
        self.elevation_map[4] += self.param.update_map_time_interval

    def input_pointcloud(
            self,
            raw_points: cp._core.core.ndarray,
            channels: List[str],
            rotation: cp._core.core.ndarray,
            translation: cp._core.core.ndarray,
            position_noise: float,
            orientation_noise: float):
        """포인트 클라우드를 입력하여 고도 지도를 업데이트하는 멤버 함수. map에서 sensor로의 rotation, translation 이용."""
        raw_points = cp.asarray(raw_points, dtype=self.data_type)
        additional_channels = channels[3:]
        raw_points = raw_points[~cp.isnan(raw_points).any(axis=1)]
        self.update_map_with_kernel(
            raw_points,
            additional_channels,
            cp.asarray(rotation, dtype=self.data_type),
            cp.asarray(translation, dtype=self.data_type),
            position_noise,
            orientation_noise)

    def input_depth(
            self,
            image: cp._core.core.ndarray,
            rotation: cp._core.core.ndarray,
            translation: cp._core.core.ndarray,
            intrinsic: cp._core.core.ndarray,
            position_noise: float,
            orientation_noise: float):
        """Depth 이미지를 입력하여 고도 지도를 업데이트하는 멤버 함수. map에서 sensor로의 rotation, translation 이용."""
        depth = cp.asarray(image, dtype=self.data_type)
        intrinsic = cp.asarray(intrinsic, dtype=self.data_type)
        pos = cp.where(depth > self.param.min_valid_distance, 1, 0)
        low = cp.where(depth < self.param.max_valid_distance, 1, 0)
        conf = cp.ones(pos.shape)
        fin = cp.isfinite(depth)
        temp = cp.maximum(cp.rint(fin + pos + conf + low - 2.6), 0)
        mask = cp.nonzero(temp)
        u = mask[1]
        v = mask[0]

        world_x = (u.astype(np.float32) - intrinsic[0, 2]) * depth[v, u] / intrinsic[0, 0]
        world_y = (v.astype(np.float32) - intrinsic[1, 2]) * depth[v, u] / intrinsic[1, 1]
        world_z = depth[v, u]
        raw_points = cp.stack((world_x, world_y, world_z), axis=1)

        additional_channels = []
        raw_points = raw_points[~cp.isnan(raw_points).any(axis=1)]
        self.update_map_with_kernel(
            raw_points,
            additional_channels,
            cp.asarray(rotation, dtype=self.data_type),
            cp.asarray(translation, dtype=self.data_type),
            position_noise,
            orientation_noise)

    def input_image(
            self,
            image: List[cp._core.core.ndarray],
            channels: List[str],
            rotation: cp._core.core.ndarray,
            translation: cp._core.core.ndarray,
            intrinsic: cp._core.core.ndarray,
            image_height: int,
            image_width: int):
        """
        RGB 혹은 Segmentation 이미지를 입력하여 고도 지도를 업데이트하는 멤버 함수.

        sensor에서 map로의 rotation, translation 이용. (역추적)
        """
        if not self.image_subscribed:
            self.compile_image_kernels()
            self.image_subscribed = True
        image = np.stack(image, axis=0)
        if len(image.shape) == 2:
            image = image[None]

        image = cp.asarray(image, dtype=self.data_type)
        intrinsic = cp.asarray(intrinsic, dtype=self.data_type)
        rotation = cp.asarray(rotation, dtype=self.data_type)
        translation = cp.asarray(translation, dtype=self.data_type)
        image_height = cp.float32(image_height)
        image_width = cp.float32(image_width)

        P = cp.asarray(
            intrinsic @ cp.concatenate([rotation, translation[:, None]], 1),
            dtype=np.float32)
        translation_camera_map = -rotation.T @ translation - self.center
        translation_camera_map = translation_camera_map.get()
        x1 = cp.uint32((self.cell_n / 2) + ((translation_camera_map[0]) / self.resolution))
        y1 = cp.uint32((self.cell_n / 2) + ((translation_camera_map[1]) / self.resolution))
        z1 = cp.float32(translation_camera_map[2])

        self.uv_correspondence *= 0
        self.valid_correspondence[:, :] = False

        with self.map_lock:
            self.image_to_map_correspondence_kernel(
                self.elevation_map,
                x1,
                y1,
                z1,
                P.reshape(-1),
                image_height,
                image_width,
                self.center,
                self.uv_correspondence,
                self.valid_correspondence,
                size=int(self.cell_n * self.cell_n))
            self.semantic_map.update_layers_image(
                image,
                channels,
                self.uv_correspondence,
                self.valid_correspondence,
                image_height,
                image_width)

    def update_normal(self, dilated_map):
        """Normal지도를 업데이트하는 멤버 함수. dilated_map는 확장된 normal 지도다."""
        with self.map_lock:
            self.normal_map *= 0.0
            self.normal_filter_kernel(
                dilated_map,
                self.elevation_map[2],
                self.normal_map,
                size=(self.cell_n * self.cell_n))

    def process_map_for_publish(self, input_map, fill_nan=False, add_z=False):
        """fill_nan, add_z, 패딩 제거를 통해 고도 지도를 post-processing하는 멤버 함수."""
        m = input_map.copy()
        if fill_nan:
            m = cp.where(self.elevation_map[2] > 0.5, m, cp.nan)
        if add_z:
            m = m + self.center[2]

        return m[1:-1, 1:-1]

    def get_elevation(self):
        return self.process_map_for_publish(self.elevation_map[0], fill_nan=True, add_z=True)

    def get_variance(self):
        return self.process_map_for_publish(self.elevation_map[1], fill_nan=False, add_z=False)

    def get_traversability(self):
        traversability = cp.where(
            (self.elevation_map[2] + self.elevation_map[6]) > 0.5,
            self.elevation_map[3].copy(),
            cp.nan)
        self.traversability_buffer[3:-3, 3:-3] = traversability[3:-3, 3:-3]
        traversability = self.traversability_buffer[1:-1, 1:-1]
        return traversability

    def get_time(self):
        return self.process_map_for_publish(self.elevation_map[4], fill_nan=False, add_z=False)

    def get_upper_bound(self):
        valid = cp.logical_or(self.elevation_map[2] > 0.5, self.elevation_map[6] > 0.5)
        upper_bound = cp.where(valid, self.elevation_map[5].copy(), cp.nan)
        upper_bound = upper_bound[1:-1, 1:-1] + self.center[2]
        return upper_bound

    def get_is_upper_bound(self):
        valid = cp.logical_or(self.elevation_map[2] > 0.5, self.elevation_map[6] > 0.5)
        is_upper_bound = cp.where(valid, self.elevation_map[6].copy(), cp.nan)
        is_upper_bound = is_upper_bound[1:-1, 1:-1]
        return is_upper_bound

    def copy_to_cpu(self, array, data, stream=None):
        """데이터를 float32로 변환하고, GPU인 경우 CPU로 로드하는 멤버 함수."""
        if type(array) == np.ndarray:
            data[...] = array.astype(np.float32)
        elif type(array) == cp.ndarray:
            if stream is not None:
                data[...] = cp.asnumpy(array.astype(np.float32), stream=stream)
            else:
                data[...] = cp.asnumpy(array.astype(np.float32))

    def exists_layer(self, name):
        if name in self.layer_names:
            return True
        elif name in self.semantic_map.layer_names:
            return True
        elif name in self.plugin_manager.layer_names:
            return True
        else:
            return False

    def get_map_with_name_ref(self, name, data):
        """특정 이름의 layer를 로드하는 멤버 함수."""
        use_stream = True
        with self.map_lock:
            if name == 'elevation':
                m = self.get_elevation()
                use_stream = False
            elif name == 'variance':
                m = self.get_variance()
            elif name == 'traversability':
                m = self.get_traversability()
            elif name == 'time':
                m = self.get_time()
            elif name == 'upper_bound':
                m = self.get_upper_bound()
            elif name == 'is_upper_bound':
                m = self.get_is_upper_bound()
            elif name == 'normal_x':
                m = self.normal_map.copy()[0, 1:-1, 1:-1]
            elif name == 'normal_y':
                m = self.normal_map.copy()[1, 1:-1, 1:-1]
            elif name == 'normal_z':
                m = self.normal_map.copy()[2, 1:-1, 1:-1]
            elif name in self.semantic_map.layer_names:
                m = self.semantic_map.get_map_with_name(name)
            elif name in self.plugin_manager.layer_names:
                self.plugin_manager.update_with_name(
                    name,
                    self.elevation_map,
                    self.layer_names,
                    self.semantic_map.semantic_map,
                    self.semantic_map.layer_names,
                    self.base_rotation,
                    self.semantic_map.elements_to_shift)
                m = self.plugin_manager.get_map_with_name(name)
                p = self.plugin_manager.get_param_with_name(name)
                m = self.process_map_for_publish(m, fill_nan=p.fill_nan, add_z=p.is_height_layer)
            else:
                print('Layer {} is not in the map'.format(name))
                return
        m = cp.flip(m, 0)
        m = cp.flip(m, 1)
        if use_stream:
            stream = cp.cuda.Stream(non_blocking=False)
        else:
            stream = None
        self.copy_to_cpu(m, data, stream=stream)
