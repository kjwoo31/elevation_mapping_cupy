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

import cupy as cp


def image_to_map_correspondence_kernel(resolution, width, height, tolerance_z_collision):
    _image_to_map_correspondence_kernel = cp.ElementwiseKernel(
        in_params='raw U map, raw U x1, raw U y1, raw U z1, raw U points, raw U image_height,' +
                  ' raw U image_width, raw U center',
        out_params='raw U uv_correspondence, raw B valid_correspondence',
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            __device__ bool is_inside_map(int x, int y) {
                return (x >= 0 && y >= 0 && x < ${width} && x < ${height});
            }
            __device__ float get_l2_distance(int x0, int y0, int x1, int y1) {
                float dx = x0 - x1;
                float dy = y0 - y1;
                return sqrt( dx * dx + dy * dy);
            }
            """
        ).substitute(width=width, height=height, resolution=resolution),
        operation=string.Template(
            """
            int cell_idx = get_map_idx(i, 0);

            // 유효한 높이값이 없으면 건너뛴다.
            if (map[get_map_idx(i, 2)] != 1){
                return;
            }

            int y0 = i % ${width};
            int x0 = i / ${width};

            float p1 = (x0 - (${width} / 2)) * ${resolution} + center[0];
            float p2 = (y0 - (${height} / 2)) * ${resolution} + center[1];
            float p3 = map[cell_idx] + center[2];

            // 3D 포인트를 이미지 평면에 투영한다.
            float u = p1 * points[0]  + p2 * points[1] + p3 * points[2] + points[3];
            float v = p1 * points[4]  + p2 * points[5] + p3 * points[6] + points[7];
            float d = p1 * points[8]  + p2 * points[9] + p3 * points[10] + points[11];

            // 이미지를 벗어나는 포인트 필터링
            if (d <= 0) {
                return;
            }
            u = u / d;
            v = v / d;
            if ((u < 0) || (v < 0) || (u >= image_width) || (v >= image_height)){
                return;
            }

            int y0_c = y0;
            int x0_c = x0;
            float total_distance = get_l2_distance(x0_c, y0_c, x1, y1);
            float z0 = map[cell_idx];
            float delta_z = z1 - z0;

            // bresenham algorithm을 사용하여 카메라 중심과 지도 사이 직선 연결
            // https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
            int dx = abs(x1 - x0);
            int sx = x0 < x1 ? 1 : -1;
            int dy = -abs(y1 - y0);
            int sy = y0 < y1 ? 1 : -1;
            int error = dx + dy;

            bool is_valid = true;

            // 직선의 모든 셀에 대해 반복한다.
            while (1) {
                if (x0 == x1 && y0 == y1) {
                    break;
                }

                // 높이값이 유효한지 확인한다.
                if (is_inside_map(x0, y0)) {
                    int idx = y0 + (x0 * ${width});
                    if (map[get_map_idx(idx, 2)]) {
                        float distance = get_l2_distance(x0_c, y0_c, x0, y0);
                        float rayheight = z0 + (distance / total_distance * delta_z);
                        if (map[idx] - ${tolerance_z_collision} > rayheight) {
                            is_valid = false;
                            break;
                        }
                    }
                }

                // 직선의 다음 셀로 이동한다.
                int e2 = 2 * error;
                if (e2 >= dy) {
                    if (x0 == x1) {
                        break;
                    }
                    error = error + dy;
                    x0 = x0 + sx;
                }
                if (e2 <= dx) {
                    if (y0 == y1) {
                        break;
                    }
                    error = error + dx;
                    y0 = y0 + sy;
                }
            }

            // 이미지와 지도 간 대응관계를 기록한다.
            uv_correspondence[get_map_idx(i, 0)] = u;
            uv_correspondence[get_map_idx(i, 1)] = v;
            valid_correspondence[get_map_idx(i, 0)] = is_valid;
            """
        ).substitute(
            height=height,
            width=width,
            resolution=resolution,
            tolerance_z_collision=tolerance_z_collision),
        name='image_to_map_correspondence_kernel')
    return _image_to_map_correspondence_kernel
