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

from abc import ABC
from abc import abstractmethod


class FusionBase(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.name = None

    @abstractmethod
    def __call__(
            self,
            points_all,
            rotation,
            translation,
            pcl_ids,
            layer_ids,
            elevation_map,
            semantic_map,
            new_map):
        pass
