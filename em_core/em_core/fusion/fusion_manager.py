#!/usr/bin/env python3
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

import importlib
import inspect
from typing import Dict

from .fusion_base import FusionBase


class FusionManager(object):

    def __init__(self, params):
        self.fusion_plugins: Dict[str, FusionBase] = {}
        self.params = params
        self.plugins = []

    def register_plugin(self, plugin):
        try:
            fusion_module = importlib.import_module('.' + plugin, package='em_core.fusion')
        except ValueError:
            raise ValueError('Plugin {} does not exist.'.format(plugin))
        for name, obj in inspect.getmembers(fusion_module):
            if inspect.isclass(obj) and issubclass(obj, FusionBase) and name != 'FusionBase':
                self.plugins.append(obj(self.params))

    def get_plugin_idx(self, name: str, data_type: str):
        name = data_type + '_' + name
        for idx, plugin in enumerate(self.plugins):
            if plugin.name == name:
                return idx
        print('[WARNING] Plugin {} is not in the list: {}'.format(name, self.plugins))
        return None

    def execute_plugin(
            self,
            name: str,
            points_all,
            rotation,
            translation,
            pcl_ids,
            layer_ids,
            elevation_map,
            semantic_map,
            new_map,
            elements_to_shift):
        idx = self.get_plugin_idx(name, 'pointcloud')
        if idx is not None:
            self.plugins[idx](
                points_all,
                rotation,
                translation,
                pcl_ids,
                layer_ids,
                elevation_map,
                semantic_map,
                new_map,
                elements_to_shift)

    def execute_image_plugin(
            self,
            name: str,
            sem_map_idx,
            image,
            class_idx,
            uv_correspondence,
            valid_correspondence,
            image_height,
            image_width,
            semantic_map,
            new_map):
        idx = self.get_plugin_idx(name, 'image')
        if idx is not None:
            self.plugins[idx](
                sem_map_idx,
                image,
                class_idx,
                uv_correspondence,
                valid_correspondence,
                image_height,
                image_width,
                semantic_map,
                new_map)
