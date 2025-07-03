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


def map_utils(
        resolution,
        width,
        height,
        sensor_noise_factor,
        min_valid_distance,
        max_height_range):
    util_preamble = string.Template(
        """
        __device__ float16 clamp(float16 x, float16 min_x, float16 max_x) {
            return max(min(x, max_x), min_x);
        }
        __device__ int get_x_idx(float16 x, float16 center) {
            int i = (x - center) / ${resolution} + 0.5 * ${width};
            return i;
        }
        __device__ int get_y_idx(float16 y, float16 center) {
            int i = (y - center) / ${resolution} + 0.5 * ${height};
            return i;
        }
        __device__ bool is_inside(int idx) {
            int idx_x = idx / ${width};
            int idx_y = idx % ${width};
            if (idx_x == 0 || idx_x == ${width} - 1) {
                return false;
            }
            if (idx_y == 0 || idx_y == ${height} - 1) {
                return false;
            }
            return true;
        }
        __device__ int get_idx(float16 x, float16 y, float16 center_x, float16 center_y) {
            int idx_x = clamp(get_x_idx(x, center_x), 0, ${width} - 1);
            int idx_y = clamp(get_y_idx(y, center_y), 0, ${height} - 1);
            return ${width} * idx_x + idx_y;
        }
        __device__ int get_map_idx(int idx, int layer_n) {
            const int layer = ${width} * ${height};
            return layer * layer_n + idx;
        }
        __device__ float transform_point(
                float16 x,
                float16 y,
                float16 z,
                float16 r0,
                float16 r1,
                float16 r2,
                float16 t) {
            return r0 * x + r1 * y + r2 * z + t;
        }
        __device__ float z_noise(float16 z){
            return ${sensor_noise_factor} * z * z;
        }

        __device__ float point_sensor_distance(
                float16 x,
                float16 y,
                float16 z,
                float16 sx,
                float16 sy,
                float16 sz) {
            float d = (x - sx) * (x - sx) + (y - sy) * (y - sy) + (z - sz) * (z - sz);
            return d;
        }

        __device__ bool is_valid(
                float16 x,
                float16 y,
                float16 z,
                float16 sx,
                float16 sy,
                float16 sz) {
            float d = point_sensor_distance(x, y, z, sx, sy, sz);
            float dxy = max(sqrt(x * x + y * y), 0.0);
            if (d < ${min_valid_distance} * ${min_valid_distance}) {
                return false;
            }
            else if (z - sz > ${max_height_range}) {
                return false;
            }
            else {
                return true;
            }
        }

        __device__ float ray_vector(
                float16 tx,
                float16 ty,
                float16 tz,
                float16 px,
                float16 py,
                float16 pz,
                float16& rx,
                float16& ry,
                float16& rz){
            float16 vx = px - tx;
            float16 vy = py - ty;
            float16 vz = pz - tz;
            float16 norm = sqrt(vx * vx + vy * vy + vz * vz);
            if (norm > 0) {
                rx = vx / norm;
                ry = vy / norm;
                rz = vz / norm;
            }
            else {
                rx = 0;
                ry = 0;
                rz = 0;
            }
            return norm;
        }

        __device__ float inner_product(
                float16 x1,
                float16 y1,
                float16 z1,
                float16 x2,
                float16 y2,
                float16 z2) {
            float product = (x1 * x2 + y1 * y2 + z1 * z2);
            return product;
       }

        """
    ).substitute(
        resolution=resolution,
        width=width,
        height=height,
        sensor_noise_factor=sensor_noise_factor,
        min_valid_distance=min_valid_distance,
        max_height_range=max_height_range)
    return util_preamble


def add_points_kernel(
        resolution,
        width,
        height,
        sensor_noise_factor,
        mahalanobis_thresh,
        outlier_variance,
        wall_num_thresh,
        max_ray_length,
        cleanup_step,
        min_valid_distance,
        max_height_range,
        cleanup_cos_thresh,
        enable_visibility_cleanup=True):
    add_points_kernel = cp.ElementwiseKernel(
        in_params='raw U center_x, raw U center_y, raw U rotation, raw U translation,' +
        'raw U norm_map',
        out_params='raw U points, raw U map, raw T newmap',
        preamble=map_utils(
            resolution,
            width,
            height,
            sensor_noise_factor,
            min_valid_distance,
            max_height_range),
        operation=string.Template(
            """
            U rx = points[i * 3];
            U ry = points[i * 3 + 1];
            U rz = points[i * 3 + 2];
            U x = transform_point(rx, ry, rz, rotation[0], rotation[1], rotation[2],
            translation[0]);
            U y = transform_point(rx, ry, rz, rotation[3], rotation[4], rotation[5],
            translation[1]);
            U z = transform_point(rx, ry, rz, rotation[6], rotation[7], rotation[8],
            translation[2]);
            U v = z_noise(rz);
            int idx = get_idx(x, y, center_x[0], center_y[0]);
            if (is_valid(x, y, z, translation[0], translation[1], translation[2])) {
                if (is_inside(idx)) {
                    U map_h = map[get_map_idx(idx, 0)];
                    U map_v = map[get_map_idx(idx, 1)];
                    U num_points = newmap[get_map_idx(idx, 4)];
                    if (abs(map_h - z) > (map_v * ${mahalanobis_thresh})) {
                        atomicAdd(&map[get_map_idx(idx, 1)], ${outlier_variance});
                    }
                    else {
                        T new_h = (map_h * v + z * map_v) / (map_v + v);
                        T new_v = (map_v * v) / (map_v + v);
                        atomicAdd(&newmap[get_map_idx(idx, 0)], new_h);
                        atomicAdd(&newmap[get_map_idx(idx, 1)], new_v);
                        atomicAdd(&newmap[get_map_idx(idx, 2)], 1.0);
                        // is Valid
                        map[get_map_idx(idx, 2)] = 1;
                        // Time layer
                        map[get_map_idx(idx, 4)] = 0.0;
                        // Upper bound
                        map[get_map_idx(idx, 5)] = new_h;
                        map[get_map_idx(idx, 6)] = 0.0;
                    }
                }
            }
            // Ray tracing
            if (${enable_visibility_cleanup}) {
                float16 ray_x, ray_y, ray_z;
                float16 ray_length = ray_vector(translation[0], translation[1], translation[2],
                x, y, z, ray_x, ray_y, ray_z);
                ray_length = min(ray_length, (float16)${max_ray_length});
                int last_nidx = -1;
                for (float16 s=${ray_step}; s < ray_length; s+=${ray_step}) {
                    // Ray마다 반복한다.
                    U nx = translation[0] + ray_x * s;
                    U ny = translation[1] + ray_y * s;
                    U nz = translation[2] + ray_z * s;
                    int nidx = get_idx(nx, ny, center_x[0], center_y[0]);
                    if (last_nidx == nidx) {continue;}  // Skip if we're still in the same cell
                    else {last_nidx = nidx;}
                    if (!is_inside(nidx)) {continue;}

                    U nmap_h = map[get_map_idx(nidx, 0)];
                    U nmap_v = map[get_map_idx(nidx, 1)];
                    U nmap_valid = map[get_map_idx(nidx, 2)];
                    // Traversability
                    U nmap_trav = map[get_map_idx(nidx, 3)];
                    // Time layer
                    U non_updated_t = map[get_map_idx(nidx, 4)];
                    // Upper bound
                    U nmap_upper = map[get_map_idx(nidx, 5)];
                    U nmap_is_upper = map[get_map_idx(nidx, 6)];

                    // Ray 길이보다 먼 포인트는 건너뛴다.
                    float16 d = (x - nx) * (x - nx) + (y - ny) * (y - ny) + (z - nz) * (z - nz);
                    if (d < 0.1 || !is_valid(
                        x, y, z, translation[0], translation[1], translation[2])) {
                        continue;
                    }

                    // 유효하지 않으면 상한을 확인하고 건너뛴다.
                    if (nmap_valid < 0.5) {
                      if (nz < nmap_upper || nmap_is_upper < 0.5) {
                        map[get_map_idx(nidx, 5)] = nz;
                        map[get_map_idx(nidx, 6)] = 1.0f;
                      }
                      continue;
                    }
                    // 최근에 엄데이트되었으면 건너뛴다.
                    if (non_updated_t < 0.5) {continue;}

                    if (nmap_h > nz + 0.01 - min(nmap_v, 1.0) * 0.05) {
                        // ray와 norm이 수직이면 건너뛴다,
                        U norm_x = norm_map[get_map_idx(nidx, 0)];
                        U norm_y = norm_map[get_map_idx(nidx, 1)];
                        U norm_z = norm_map[get_map_idx(nidx, 2)];
                        float product = inner_product(ray_x, ray_y, ray_z, norm_x, norm_y, norm_z);
                        if (fabs(product) < ${cleanup_cos_thresh}) {continue;}
                        U num_points = newmap[get_map_idx(nidx, 3)];
                        if (num_points > ${wall_num_thresh} && non_updated_t < 1.0) {continue;}

                        // ray가 관통하면 업데이트한다.
                        atomicAdd(&map[get_map_idx(nidx, 2)],
                                  -${cleanup_step}/(ray_length / ${max_ray_length}));
                        atomicAdd(&map[get_map_idx(nidx, 1)], ${outlier_variance});

                        // Upper bound 확인
                        if (nz < nmap_upper || nmap_is_upper < 0.5) {
                            map[get_map_idx(nidx, 5)] = nz;
                            map[get_map_idx(nidx, 6)] = 1.0f;
                        }
                    }
                }
            }
            points[i * 3]= idx;
            points[i * 3 + 1] = is_valid(x, y, z, translation[0], translation[1], translation[2]);
            points[i * 3 + 2] = is_inside(idx);
            """
        ).substitute(
            mahalanobis_thresh=mahalanobis_thresh,
            outlier_variance=outlier_variance,
            wall_num_thresh=wall_num_thresh,
            ray_step=resolution / 2 ** 0.5,
            max_ray_length=max_ray_length,
            cleanup_step=cleanup_step,
            cleanup_cos_thresh=cleanup_cos_thresh,
            enable_visibility_cleanup=int(enable_visibility_cleanup)
        ),
        name='add_points_kernel')
    return add_points_kernel


def error_counting_kernel(
        resolution,
        width,
        height,
        sensor_noise_factor,
        mahalanobis_thresh,
        outlier_variance,
        traversability_inlier,
        min_valid_distance,
        max_height_range):
    error_counting_kernel = cp.ElementwiseKernel(
        in_params='raw U map, raw U points, raw U center_x, raw U center_y, raw U rotation,' +
        'raw U translation',
        out_params='raw U newmap, raw T error, raw T error_count',
        preamble=map_utils(
            resolution,
            width,
            height,
            sensor_noise_factor,
            min_valid_distance,
            max_height_range
        ),
        operation=string.Template(
            """
            U rx = points[i * 3];
            U ry = points[i * 3 + 1];
            U rz = points[i * 3 + 2];
            U x = transform_point(rx, ry, rz, rotation[0], rotation[1], rotation[2],
                translation[0]);
            U y = transform_point(rx, ry, rz, rotation[3], rotation[4], rotation[5],
                translation[1]);
            U z = transform_point(rx, ry, rz, rotation[6], rotation[7], rotation[8],
                translation[2]);
            U v = z_noise(rz);
            if (!is_valid(x, y, z, translation[0], translation[1], translation[2])) {return;}
            int idx = get_idx(x, y, center_x[0], center_y[0]);
            if (!is_inside(idx)) {
                return;
            }
            U map_h = map[get_map_idx(idx, 0)];
            U map_v = map[get_map_idx(idx, 1)];
            U map_valid = map[get_map_idx(idx, 2)];
            U map_t = map[get_map_idx(idx, 3)];
            if (map_valid > 0.5 && (abs(map_h - z) < (map_v * ${mahalanobis_thresh})) &&
                map_v < ${outlier_variance} / 2.0 && map_t > ${traversability_inlier}) {
                    T e = z - map_h;
                    atomicAdd(&error[0], e);
                    atomicAdd(&error_count[0], 1);
                    atomicAdd(&newmap[get_map_idx(idx, 3)], 1.0);
            }
            atomicAdd(&newmap[get_map_idx(idx, 4)], 1.0);
            """
        ).substitute(
            mahalanobis_thresh=mahalanobis_thresh,
            outlier_variance=outlier_variance,
            traversability_inlier=traversability_inlier
        ),
        name='error_counting_kernel')
    return error_counting_kernel


def average_map_kernel(width, height, max_variance, initial_variance):
    average_map_kernel = cp.ElementwiseKernel(
        in_params='raw U newmap',
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
            U h = map[get_map_idx(i, 0)];
            U v = map[get_map_idx(i, 1)];
            U valid = map[get_map_idx(i, 2)];
            U new_h = newmap[get_map_idx(i, 0)];
            U new_v = newmap[get_map_idx(i, 1)];
            U new_count = newmap[get_map_idx(i, 2)];
            if (new_count > 0) {
                if (new_v / new_count > ${max_variance}) {
                    map[get_map_idx(i, 0)] = 0;
                    map[get_map_idx(i, 1)] = ${initial_variance};
                    map[get_map_idx(i, 2)] = 0;
                }
                else {
                    map[get_map_idx(i, 0)] = new_h / new_count;
                    map[get_map_idx(i, 1)] = new_v / new_count;
                    map[get_map_idx(i, 2)] = 1;
                }
            }
            if (valid < 0.5) {
                map[get_map_idx(i, 0)] = 0;
                map[get_map_idx(i, 1)] = ${initial_variance};
                map[get_map_idx(i, 2)] = 0;
            }
            """
        ).substitute(max_variance=max_variance, initial_variance=initial_variance),
        name='average_map_kernel')
    return average_map_kernel


def dilation_filter_kernel(width, height, dilation_size):
    """Traversability 업데이트 이전에 dilation을 수행하는 함수."""
    dilation_filter_kernel = cp.ElementwiseKernel(
        in_params='raw U map, raw U mask',
        out_params='raw U newmap, raw U newmask',
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }

            __device__ int get_relative_map_idx(int idx, int dx, int dy, int layer_n) {
                const int layer = ${width} * ${height};
                const int relative_idx = idx + ${width} * dy + dx;
                return layer * layer_n + relative_idx;
            }
            __device__ bool is_inside(int idx) {
                int idx_x = idx / ${width};
                int idx_y = idx % ${width};
                if (idx_x <= 0 || idx_x >= ${width} - 1) {
                    return false;
                }
                if (idx_y <= 0 || idx_y >= ${height} - 1) {
                    return false;
                }
                return true;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            U h = map[get_map_idx(i, 0)];
            U valid = mask[get_map_idx(i, 0)];
            newmap[get_map_idx(i, 0)] = h;
            if (valid < 0.5) {
                U distance = 100;
                U near_value = 0;
                for (int dy = -${dilation_size}; dy <= ${dilation_size}; dy++) {
                    for (int dx = -${dilation_size}; dx <= ${dilation_size}; dx++) {
                        int idx = get_relative_map_idx(i, dx, dy, 0);
                        if (!is_inside(idx)) {continue;}
                        U valid = mask[idx];
                        if(valid > 0.5 && dx + dy < distance) {
                            distance = dx + dy;
                            near_value = map[idx];
                        }
                    }
                }
                if(distance < 100) {
                    newmap[get_map_idx(i, 0)] = near_value;
                    newmask[get_map_idx(i, 0)] = 1.0;
                }
            }
            """
        ).substitute(dilation_size=dilation_size),
        name='dilation_filter_kernel')
    return dilation_filter_kernel


def normal_filter_kernel(width, height, resolution):
    normal_filter_kernel = cp.ElementwiseKernel(
        in_params='raw U map, raw U mask',
        out_params='raw U newmap',
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }

            __device__ int get_relative_map_idx(int idx, int dx, int dy, int layer_n) {
                const int layer = ${width} * ${height};
                const int relative_idx = idx + ${width} * dy + dx;
                return layer * layer_n + relative_idx;
            }
            __device__ bool is_inside(int idx) {
                int idx_x = idx / ${width};
                int idx_y = idx % ${width};
                if (idx_x <= 0 || idx_x >= ${width} - 1) {
                    return false;
                }
                if (idx_y <= 0 || idx_y >= ${height} - 1) {
                    return false;
                }
                return true;
            }
            __device__ float resolution() {
                return ${resolution};
            }
            """
        ).substitute(width=width, height=height, resolution=resolution),
        operation=string.Template(
            """
            U h = map[get_map_idx(i, 0)];
            U valid = mask[get_map_idx(i, 0)];
            if (valid > 0.5) {
                int idx_x = get_relative_map_idx(i, 1, 0, 0);
                int idx_y = get_relative_map_idx(i, 0, 1, 0);
                if (!is_inside(idx_x) || !is_inside(idx_y)) {return;}
                float dzdx = (map[idx_x] - h);
                float dzdy = (map[idx_y] - h);
                float nx = -dzdy / resolution();
                float ny = -dzdx / resolution();
                float nz = 1;
                float norm = sqrt((nx * nx) + (ny * ny) + 1);
                newmap[get_map_idx(i, 0)] = nx / norm;
                newmap[get_map_idx(i, 1)] = ny / norm;
                newmap[get_map_idx(i, 2)] = nz / norm;
            }
            """
        ).substitute(),
        name='normal_filter_kernel')
    return normal_filter_kernel
