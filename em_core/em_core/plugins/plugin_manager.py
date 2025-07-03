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
import importlib
import inspect
from inspect import signature
from typing import Dict
from typing import List

import cupy as cp
from ruamel.yaml import YAML

from .plugin_base import PluginBase


@dataclass
class PluginParams:
    name: str
    layer_name: str
    fill_nan: bool = False  # 유효하지 않은 셀에 NaN 채우기
    is_height_layer: bool = False  # 높이 layer 여부


class PluginManager(object):

    def __init__(self, cell_n: int):
        self.cell_n = cell_n

    def init(self, plugin_params: List[PluginParams], extra_params: List[Dict]):
        self.plugin_params = plugin_params

        self.plugins = []
        for param, extra_param in zip(plugin_params, extra_params):
            module = importlib.import_module('.' + param.name, package='em_core.plugins')
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, PluginBase) and name != 'PluginBase':
                    extra_param['cell_n'] = self.cell_n
                    self.plugins.append(obj(**extra_param))
        self.layers = cp.zeros((len(self.plugins), self.cell_n, self.cell_n), dtype=cp.float32)
        self.layer_names = self.get_layer_names()
        self.plugin_names = self.get_plugin_names()

    def load_plugin_settings(self, file_path: str):
        print('Start loading plugins...')
        cfg = YAML().load(open(file_path, 'r'))
        plugin_params = []
        extra_params = []
        if cfg is not None:
            for k, v in cfg.items():
                if v['enable']:
                    plugin_params.append(
                        PluginParams(
                            name=k if 'type' not in v else v['type'],
                            layer_name=v['layer_name'],
                            fill_nan=v['fill_nan'],
                            is_height_layer=v['is_height_layer'],
                        )
                    )
                    extra_params.append(v['extra_params'])
            self.init(plugin_params, extra_params)
            print('Loaded plugins are ', *self.plugin_names)

    def get_layer_names(self):
        names = []
        for obj in self.plugin_params:
            names.append(obj.layer_name)
        return names

    def get_plugin_names(self):
        names = []
        for obj in self.plugin_params:
            names.append(obj.name)
        return names

    def get_plugin_index_with_name(self, name: str) -> int:
        try:
            idx = self.plugin_names.index(name)
            return idx
        except ValueError as e:
            print('Error with plugin {}: {}'.format(name, e))
            return None

    def get_layer_index_with_name(self, name: str) -> int:
        try:
            idx = self.layer_names.index(name)
            return idx
        except ValueError as e:
            print('Error with layer {}: {}'.format(name, e))
            return None

    def update_with_name(
            self,
            name: str,
            elevation_map: cp.ndarray,
            layer_names: List[str],
            semantic_map=None,
            semantic_params=None,
            rotation=None,
            elements_to_shift={}):
        idx = self.get_layer_index_with_name(name)
        if idx is not None and idx < len(self.plugins):
            n_param = len(signature(self.plugins[idx]).parameters)
            if n_param == 5:
                self.layers[idx] = self.plugins[idx](
                    elevation_map,
                    layer_names,
                    self.layers,
                    self.layer_names)
            elif n_param == 7:
                self.layers[idx] = self.plugins[idx](
                    elevation_map,
                    layer_names,
                    self.layers,
                    self.layer_names,
                    semantic_map,
                    semantic_params)
            elif n_param == 8:
                self.layers[idx] = self.plugins[idx](
                    elevation_map,
                    layer_names,
                    self.layers,
                    self.layer_names,
                    semantic_map,
                    semantic_params,
                    rotation)
            else:
                self.layers[idx] = self.plugins[idx](
                    elevation_map,
                    layer_names,
                    self.layers,
                    self.layer_names,
                    semantic_map,
                    semantic_params,
                    rotation,
                    elements_to_shift)

    def get_map_with_name(self, name: str) -> cp.ndarray:
        idx = self.get_layer_index_with_name(name)
        if idx is not None:
            return self.layers[idx]

    def get_param_with_name(self, name: str) -> PluginParams:
        idx = self.get_layer_index_with_name(name)
        if idx is not None:
            return self.plugin_params[idx]
