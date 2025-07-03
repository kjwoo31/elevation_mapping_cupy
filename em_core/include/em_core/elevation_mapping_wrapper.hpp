// Copyright 2024 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// author : Jinwoo Kim, original author: Takahiro Miki

#ifndef EM_CORE__ELEVATION_MAPPING_WRAPPER_HPP_
#define EM_CORE__ELEVATION_MAPPING_WRAPPER_HPP_

#include <Eigen/Dense>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include "rclcpp/rclcpp.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "grid_map_msgs/msg/grid_map.hpp"
#include "grid_map_ros/grid_map_ros.hpp"

namespace em_core
{

class ElevationMappingWrapper
{
public:
  ElevationMappingWrapper();

  using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ColMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

  void initialize(rclcpp::Node * node);

  // 포인트클라우드 정보를 elevation_mapping.py에 전달하는 멤버 함수
  void input_pcd(
    const RowMatrixXd & points,
    const std::vector<std::string> & channels,
    const RowMatrixXd & rotation,
    const Eigen::VectorXd & translation,
    const double position_noise,
    const double orientation_noise);

  // Depth 이미지 정보를 elevation_mapping.py에 전달하는 멤버 함수
  void input_depth(
    const ColMatrixXf & image,
    const RowMatrixXd & rotation,
    const Eigen::VectorXd & translation,
    const RowMatrixXd & camera_matrix,
    const double position_noise,
    const double orientation_noise);

  // RGB 혹은 Segmentation 이미지를 elevation_mapping.py에 전달하는 멤버 함수
  void input_image(
    const std::vector<ColMatrixXf> & multichannel_image,
    const std::vector<std::string> & channels,
    const RowMatrixXd & rotation,
    const Eigen::VectorXd & translation,
    const RowMatrixXd & camera_matrix,
    int height,
    int width);

  void move_to(const Eigen::VectorXd & position, const RowMatrixXd & rotation);
  void clear();
  void update_variance();
  void update_time();

  // 특정 레이어가 존재하는지 확인하는 멤버 함수
  bool exists_layer(const std::string & layer_name);

  // 특정 레이어의 지도를 가져오는 멤버 함수
  void get_layer_data(const std::string & layer_name, RowMatrixXf & map);

  // 모든 레이어의 지도를 가져오는 멤버 함수
  void get_grid_map(grid_map::GridMap & grid_map, const std::vector<std::string> & layer_names);

private:
  void set_parameters(rclcpp::Node * node);
  pybind11::object map_;
  pybind11::object param_;
  double resolution_;
  double map_length_;
  int map_n_;
};

}  // namespace em_core

#endif  // EM_CORE__ELEVATION_MAPPING_WRAPPER_HPP_
