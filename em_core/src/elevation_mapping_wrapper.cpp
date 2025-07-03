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

#include "em_core/elevation_mapping_wrapper.hpp"

#include <string>
#include <memory>
#include <vector>

namespace em_core
{

ElevationMappingWrapper::ElevationMappingWrapper() {}

void ElevationMappingWrapper::initialize(rclcpp::Node * node)
{
  // parmeters.py의 매개변수를 불러오고 elevation_mapping.py에 초기 지도를 설정한다.
  /*
  매개변수
  ----------
    resolution: m 단위 지도 해상도
      (Default: ``0.04``)
    additional_layers: 지도 추가 layer
      (Default: ``['color']``)
    fusion_algorithms: 이미지 및 포인트클라우드의 지도 fusion 알고리즘 list
      (Default: ``[ 'image_color', 'image_exponential', 'pointcloud_class_average', 'pointcloud_color']``)
    pointcloud_channel_fusions: 포인트클라우드 fusion 방법
      (Default: ``{'rgb': 'color', 'default': 'class_average'}``)
    image_channel_fusions: 이미지 fusion 방법
      (Default: ``{'rgb': 'color', 'default': 'exponential'}``)
    data_type: 지도 데이터 타입
      (Default: ``np.float32``)
    average_weight: average fusion 가중치
      (Default: ``0.5``)
    map_length: m 단위 지도 길이
      (Default: ``8.0``)
    sensor_noise_factor: sensor_noise_factor*(센서로부터 포인트의 거리)^2으로 포인트 오차를 계산한다.
      (Default: ``0.05``)
    mahalanobis_thresh: outlier를 판단하는 Mahalanobis distance 기준
      (Default: ``2.0``)
    outlier_variance: outlier일 경우, cell에 더하는 값
      (Default: ``0.01``)
    drift_compensation_variance_inlier: 높이 오차 보정에 사용하는 분산값 기준 (아래값만 사용)
      (Default: ``0.1``)
    time_variance: update_variance를 호출할 때마다 더하는 시간 분산값
      (Default: ``0.01``)
    update_map_time_interval: 시간 layer 업데이트 주기
      (Default: ``0.1``)
    max_variance: 각 셀의 최대 분산값
      (Default: ``1.0``)
    dilation_size: traversability filter 이전에 사용하는 dilation filter 크기
      (Default: ``2``)
    drift_compensation_alpha: (오차)*alpha만큼 높이 오차 보정을 진행한다. alpha를 줄여 부드러운 보정이 가능하다.
      (Default: ``1.0``)
    traversability_inlier: 높이 오차 보정을 시작하는 traversability 기준
      (Default: ``0.1``)
    wall_num_thresh: 벽을 날카롭게 나타내기 위해 현재 높이 대신 최대 높이를 활용하는 포인트 개수 기준
      (Default: ``100``)
    min_height_drift_count: 높이 오차 보정을 시작하는 포인트 개수 기준
      (Default: ``100``)
    max_ray_length: Ray tracing 최대 길이
      (Default: ``2.0``)
    cleanup_step: Ray tracing을 적용하는 길이 단위
      (Default: ``0.01``)
    cleanup_cos_thresh: Ray tracing을 적용하는 내적값 기준
      (Default: ``0.5``)
    min_valid_distance: 로봇으로부터 거리가 짧은 포인트 필터링 범위
      (Default: ``0.3``)
    max_valid_distance: Depth 이미지 최대 허용 거리 범위
      (Default: ``10.0``)
    max_height_range: 포인트 최대 높이 범위. (천장 필터링)
      (Default: ``1.0``)
    max_drift: 최대 높이 오차 보정값
      (Default: ``0.10``)
    enable_drift_compensation: 높이 오차 보정 활성화
      (Default: ``True``)
    enable_visibility_cleanup: Ray tracing 활성화
      (Default: ``True``)
    position_noise_thresh: 높이 오차 보정을 시작하는 위치 오차 기준
      (Default: ``0.1``)
    orientation_noise_thresh: 높이 오차 보정을 시작하는 회전 오차 기준
      (Default: ``0.1``)
    plugin_config_file: 플러그인 구성 파일
      (Default: ``'config/plugin_config.yaml'``)
    weight_file: traversability filter를 위한 가중치 파일
      (Default: ``'config/weights.dat'``)
    initial_variance: 각 셀의 초기 분산값
      (Default: ``10.0``)
    w1: 1번째 layer 가중치
      (Default: ``np.zeros((4, 1, 3, 3))``)
    w2: 2번째 layer 가중치
      (Default: ``np.zeros((4, 1, 3, 3))``)
    w3: 3번째 layer 가중치
      (Default: ``np.zeros((4, 1, 3, 3))``)
    w_out: 출력 layer 가중치
      (Default: ``np.zeros((1, 12, 1, 1))``)
    true_map_length: m 단위 테두리를 제외한 지도 길이
      (Default: ``None``)
    cell_n: 테두리를 포함한 지도 길이 셀 개수
      (Default: ``None``)
    true_cell_n: 테두리를 제외한 지도 길이 셀 개수
      (Default: ``None``)
  */
  auto elevation_mapping = pybind11::module::import("em_core.elevation_mapping");
  auto parameter = pybind11::module::import("em_core.parameter");
  param_ = parameter.attr("Parameter")();
  pybind11::list paramNames = param_.attr("get_names")();
  pybind11::list paramTypes = param_.attr("get_types")();
  pybind11::gil_scoped_acquire acquire;
  for (int i = 0; i < static_cast<int>(paramNames.size()); ++i) {
    std::string type = pybind11::cast<std::string>(paramTypes[i]);
    std::string name = pybind11::cast<std::string>(paramNames[i]);
    if (!node->has_parameter(name)) {node->declare_parameter(name);}
    if (type == "float") {
      try {
        double param = node->get_parameter(name).as_double();
        param_.attr("set_value")(name, param);
      } catch (rclcpp::exceptions::ParameterNotDeclaredException & e) {
        continue;
      } catch (rclcpp::ParameterTypeException & e) {
        RCLCPP_INFO_STREAM(
          node->get_logger(), "Parameter " << name << " should be float. Using default value");
      }
    } else if (type == "str") {
      try {
        std::string param = node->get_parameter(name).as_string();
        param_.attr("set_value")(name, param);
      } catch (rclcpp::exceptions::ParameterNotDeclaredException & e) {
        continue;
      }
    } else if (type == "bool") {
      try {
        bool param = node->get_parameter(name).as_bool();
        param_.attr("set_value")(name, param);
      } catch (rclcpp::exceptions::ParameterNotDeclaredException & e) {
        continue;
      }
    } else if (type == "int") {
      try {
        int param = node->get_parameter(name).as_int();
        param_.attr("set_value")(name, param);
      } catch (rclcpp::exceptions::ParameterNotDeclaredException & e) {
        continue;
      }
    }
  }

  // parameter.py에서 self.cell_n, self.true_cell_n, self.true_map_length 업데이트
  param_.attr("update")();
  resolution_ = pybind11::cast<float>(param_.attr("get_value")("resolution"));
  map_length_ = pybind11::cast<float>(param_.attr("get_value")("true_map_length"));
  map_n_ = pybind11::cast<int>(param_.attr("get_value")("true_cell_n"));
  map_ = elevation_mapping.attr("ElevationMap")(param_);
}

void ElevationMappingWrapper::input_pcd(
  const RowMatrixXd & points,
  const std::vector<std::string> & channels,
  const RowMatrixXd & rotation,
  const Eigen::VectorXd & translation,
  const double position_noise,
  const double orientation_noise)
{
  pybind11::gil_scoped_acquire acquire;
  map_.attr("input_pointcloud")(
    Eigen::Ref<const RowMatrixXd>(points),
    channels,
    Eigen::Ref<const RowMatrixXd>(rotation),
    Eigen::Ref<const Eigen::VectorXd>(translation),
    position_noise,
    orientation_noise);
}

void ElevationMappingWrapper::input_depth(
  const ColMatrixXf & image,
  const RowMatrixXd & rotation,
  const Eigen::VectorXd & translation,
  const RowMatrixXd & camera_matrix,
  const double position_noise,
  const double orientation_noise)
{
  pybind11::gil_scoped_acquire acquire;
  map_.attr("input_depth")(
    image,
    Eigen::Ref<const RowMatrixXd>(rotation),
    Eigen::Ref<const Eigen::VectorXd>(translation),
    Eigen::Ref<const RowMatrixXd>(camera_matrix),
    position_noise,
    orientation_noise);
}

void ElevationMappingWrapper::input_image(
  const std::vector<ColMatrixXf> & multichannel_image,
  const std::vector<std::string> & channels,
  const RowMatrixXd & rotation,
  const Eigen::VectorXd & translation,
  const RowMatrixXd & camera_matrix,
  int height,
  int width)
{
  pybind11::gil_scoped_acquire acquire;
  map_.attr("input_image")(
    multichannel_image,
    channels,
    Eigen::Ref<const RowMatrixXd>(rotation),
    Eigen::Ref<const Eigen::VectorXd>(translation),
    Eigen::Ref<const RowMatrixXd>(camera_matrix),
    height,
    width);
}

void ElevationMappingWrapper::move_to(
  const Eigen::VectorXd & position,
  const RowMatrixXd & rotation)
{
  pybind11::gil_scoped_acquire acquire;
  map_.attr("move_to")(
    Eigen::Ref<const Eigen::VectorXd>(position),
    Eigen::Ref<const RowMatrixXd>(rotation));
}

void ElevationMappingWrapper::clear()
{
  pybind11::gil_scoped_acquire acquire;
  map_.attr("clear")();
}

bool ElevationMappingWrapper::exists_layer(const std::string & layer_name)
{
  pybind11::gil_scoped_acquire acquire;
  return pybind11::cast<bool>(map_.attr("exists_layer")(layer_name));
}

void ElevationMappingWrapper::get_layer_data(const std::string & layer_name, RowMatrixXf & map)
{
  pybind11::gil_scoped_acquire acquire;
  map = RowMatrixXf(map_n_, map_n_);
  map_.attr("get_map_with_name_ref")(layer_name, Eigen::Ref<RowMatrixXf>(map));
}

void ElevationMappingWrapper::get_grid_map(
  grid_map::GridMap & grid_map,
  const std::vector<std::string> & requestlayer_names)
{
  RowMatrixXd pos(1, 3);
  pybind11::gil_scoped_acquire acquire;
  map_.attr("get_map_center_position")(Eigen::Ref<RowMatrixXd>(pos));
  grid_map::Position position(pos(0, 0), pos(0, 1));
  grid_map::Length length(map_length_, map_length_);
  grid_map.setGeometry(length, resolution_, position);

  std::vector<std::string> layer_names = requestlayer_names;
  for (const auto & layer_name : layer_names) {
    bool exists = map_.attr("exists_layer")(layer_name).cast<bool>();
    if (exists) {
      RowMatrixXf map(map_n_, map_n_);
      map_.attr("get_map_with_name_ref")(layer_name, Eigen::Ref<RowMatrixXf>(map));
      grid_map.add(layer_name, map);
    }
  }
}

void ElevationMappingWrapper::update_variance()
{
  pybind11::gil_scoped_acquire acquire;
  map_.attr("update_variance")();
}

void ElevationMappingWrapper::update_time()
{
  pybind11::gil_scoped_acquire acquire;
  map_.attr("update_time")();
}

}  // namespace em_core
