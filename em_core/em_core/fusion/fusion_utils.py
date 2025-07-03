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

import string

import cupy as cp


def color_correspondences_to_map_kernel(resolution, width, height):
    """이미지와 지도 간 대응 관계에 따라 RGB 값을 지도에 저장하는 함수."""
    color_correspondences_to_map_kernel = cp.ElementwiseKernel(
        in_params='raw U sem_map, raw U map_idx, raw U image_rgb,' +
                  'raw U uv_correspondence, raw B valid_correspondence,' +
                  'raw U image_height, raw U image_width',
        out_params='raw U new_sem_map',
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            int cell_idx = get_map_idx(i, 0);
            if (valid_correspondence[cell_idx]){
                int cell_idx_2 = get_map_idx(i, 1);

                int idx_red = int(uv_correspondence[cell_idx]) +
                    int(uv_correspondence[cell_idx_2]) * image_width;
                int idx_green = image_width * image_height + idx_red;
                int idx_blue = image_width * image_height * 2 + idx_red;

                unsigned int r = image_rgb[idx_red];
                unsigned int g = image_rgb[idx_green];
                unsigned int b = image_rgb[idx_blue];

                unsigned int rgb = (r << 16) + (g << 8) + b;
                float rgb_ = __uint_as_float(rgb);
                new_sem_map[get_map_idx(i, map_idx)] = rgb_;
            } else {
                new_sem_map[get_map_idx(i, map_idx)] = sem_map[get_map_idx(i, map_idx)];
            }
            """
        ).substitute(),
        name='color_correspondences_to_map_kernel')
    return color_correspondences_to_map_kernel


def exponential_correspondences_to_map_kernel(resolution, width, height, alpha):
    """이미지와 지도 간 대응 관계에 따라 alpha 계수만큼 지도를 업데이트하는 함수."""
    exponential_correspondences_to_map_kernel = cp.ElementwiseKernel(
        in_params='raw U sem_map, raw U map_idx, raw U image_mono, raw U uv_correspondence,' +
                  'raw B valid_correspondence, raw U image_height, raw U image_width',
        out_params='raw U new_sem_map',
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            int cell_idx = get_map_idx(i, 0);
            if (valid_correspondence[cell_idx]) {
                int cell_idx_2 = get_map_idx(i, 1);
                int idx = int(uv_correspondence[cell_idx]) + int(uv_correspondence[cell_idx_2]) *
                    image_width;
                new_sem_map[get_map_idx(i, map_idx)] = sem_map[get_map_idx(i, map_idx)](1 -
                    ${alpha}) + ${alpha} * image_mono[idx];
            } else {
                new_sem_map[get_map_idx(i, map_idx)] = sem_map[get_map_idx(i, map_idx)];
            }

            """
        ).substitute(alpha=alpha),
        name='exponential_correspondences_to_map_kernel')
    return exponential_correspondences_to_map_kernel


def sum_kernel(resolution, width, height):
    """포인트클라우드 값을 지도에 저장하는 함수."""
    sum_kernel = cp.ElementwiseKernel(
        in_params='raw U points, raw U rotation, raw U translation,'
        + 'raw W pcl_layer, raw W map_lay, raw W pcl_channels',
        out_params='raw U map, raw U newmap',
        preamble=string.Template(
            """
                __device__ int get_map_idx(int idx, int layer_n) {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }
            """
        ).substitute(resolution=resolution, width=width, height=height),
        operation=string.Template(
            """
            U id = floorf(i / pcl_channels[1]);
            int layer = i % pcl_channels[1];
            U idx = points[id * pcl_channels[0]];
            U valid = points[id * pcl_channels[0] + 1];
            U inside = points[id * pcl_channels[0] + 2];
            if (valid) {
                if (inside) {
                    U feat = points[id * pcl_channels[0] + pcl_layer[layer]];
                    atomicAdd(&newmap[get_map_idx(idx, map_lay[layer])], feat);
                }
            }
            """
        ).substitute(),
        name='sum_kernel')
    return sum_kernel


def class_average_kernel(width, height, alpha):
    """포인트클라우드 id에 따라 각각 평균 내어 지도에 저장하는 함수."""
    class_average_kernel = cp.ElementwiseKernel(
        in_params='raw V newmap, raw W pcl_layer, raw W map_lay,' +
                  'raw W pcl_channels, raw U new_elevation_map',
        out_params='raw U map',
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            U id = floorf(i / pcl_channels[1]);
            int layer = i % pcl_channels[1];
            U count = new_elevation_map[get_map_idx(id, 2)];
            if (count > 0) {
                U prev_val = map[get_map_idx(id,  map_lay[layer])];
                if (prev_val==0) {
                    U val = newmap[get_map_idx(id, map_lay[layer])] / (count);
                    map[get_map_idx(id,  map_lay[layer])] = val;
                }
                else {
                    U val = ${alpha} * prev_val +
                        (1 - ${alpha}) * newmap[get_map_idx(id, map_lay[layer])] / (count);
                    map[get_map_idx(id,  map_lay[layer])] = val;
                }
            }
            """
        ).substitute(alpha=alpha),
        name='class_average_kernel')
    return class_average_kernel


def add_color_kernel(width, height):
    """포인트클라우드 RGB 값을 지도에 저장하는 함수."""
    add_color_kernel = cp.ElementwiseKernel(
        in_params='raw T points, raw U R, raw U t, raw W pcl_layer, raw W map_lay,' +
        'raw W pcl_channels',
        out_params='raw V color_map',
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            __device__ unsigned int get_r(unsigned int color) {
                unsigned int red = 0xFF0000;
                unsigned int reds = (color & red) >> 16;
                return reds;
            }
            __device__ unsigned int get_g(unsigned int color) {
                unsigned int green = 0xFF00;
                unsigned int greens = (color & green) >> 8;
                return greens;
            }
            __device__ unsigned int get_b(unsigned int color) {
                unsigned int blue = 0xFF;
                unsigned int blues = (color & blue);
                return blues;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            U id = floorf(i / pcl_channels[1]);
            int layer = i % pcl_channels[1];
            U idx = points[id * pcl_channels[0]];
            U valid = points[id * pcl_channels[0] + 1];
            U inside = points[id * pcl_channels[0] + 2];
            if (valid && inside) {
                unsigned int color = __float_as_uint(points[id * pcl_channels[0] +
                    pcl_chan[layer]]);
                atomicAdd(&color_map[get_map_idx(idx, layer * 3)], get_r(color));
                atomicAdd(&color_map[get_map_idx(idx, layer * 3 + 1)], get_g(color));
                atomicAdd(&color_map[get_map_idx(idx, layer * 3 + 2)], get_b(color));
                atomicAdd(&color_map[get_map_idx(idx, pcl_channels[1] * 3)], 1);
            }
            """
        ).substitute(width=width),
        name='add_color_kernel')
    return add_color_kernel


def color_average_kernel(width, height):
    """포인트클라우드 RGB 값을 평균 내어 지도에 저장하는 함수."""
    color_average_kernel = cp.ElementwiseKernel(
        in_params='raw V color_map, raw W pcl_layer, raw W map_lay, raw W pcl_channels',
        out_params='raw U map',
        preamble=string.Template(
            """
            __device__ int get_map_idx (int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            __device__ unsigned int get_r (unsigned int color) {
                unsigned int red = 0xFF0000;
                unsigned int reds = (color & red) >> 16;
                return reds;
            }
            __device__ unsigned int get_g (unsigned int color) {
                unsigned int green = 0xFF00;
                unsigned int greens = (color & green) >> 8;
                return greens;
            }
            __device__ unsigned int get_b (unsigned int color) {
                unsigned int blue = 0xFF;
                unsigned int blues = (color & blue);
                return blues;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            U id = floorf(i / pcl_channels[1]);
            int layer = i % pcl_channels[1];
            unsigned int count = color_map[get_map_idx(id, pcl_channels[1] * 3)];
            if (count > 0) {
                    unsigned int r = color_map[get_map_idx(id, layer * 3)] / (1 * count);
                    unsigned int g = color_map[get_map_idx(id, layer * 3 + 1)] / (1 * count);
                    unsigned int b = color_map[get_map_idx(id, layer * 3 + 2)] / (1 * count);
                    unsigned int rgb = (r << 16) + (g << 8) + b;
                    float rgb_ = __uint_as_float(rgb);
                    map[get_map_idx(id, map_lay[layer])] = rgb_;
            }
            """
        ).substitute(),
        name='color_average_kernel')
    return color_average_kernel
