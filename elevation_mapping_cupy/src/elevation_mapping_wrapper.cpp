//
// Copyright (c) 2022, Takahiro Miki. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#include "elevation_mapping_cupy/elevation_mapping_wrapper.hpp"

// Pybind
#include <pybind11/eigen.h>

// ROS
#include <ros/package.h>

#include <utility>

namespace elevation_mapping_cupy {

ElevationMappingWrapper::ElevationMappingWrapper() {}

void ElevationMappingWrapper::initialize(ros::NodeHandle& nh) {
  // Add the elevation_mapping_cupy path to sys.path
  auto threading = py::module::import("threading");
  py::gil_scoped_acquire acquire;

  auto sys = py::module::import("sys");
  auto path = sys.attr("path");
  std::string module_path = ros::package::getPath("elevation_mapping_cupy");
  module_path = module_path + "/script";
  path.attr("insert")(0, module_path);

  auto elevation_mapping = py::module::import("elevation_mapping_cupy.elevation_mapping");
  auto parameter = py::module::import("elevation_mapping_cupy.parameter");
  param_ = parameter.attr("Parameter")();
  setParameters(nh);
  map_ = elevation_mapping.attr("ElevationMap")(param_);
}

/**
 *  Load ros parameters into Parameter class.
 *  Search for the same name within the name space.
 */
void ElevationMappingWrapper::setParameters(ros::NodeHandle& nh) {
  // Get all parameters names and types.
  py::list paramNames = param_.attr("get_names")();
  py::list paramTypes = param_.attr("get_types")();
  py::gil_scoped_acquire acquire;

  // Try to find the parameter in the ros parameter server.
  // If there was a parameter, set it to the Parameter variable.
  for (int i = 0; i < paramNames.size(); i++) {
    std::string type = py::cast<std::string>(paramTypes[i]);
    std::string name = py::cast<std::string>(paramNames[i]);
    if (type == "float") {
      float param;
      if (nh.getParam(name, param)) {
        param_.attr("set_value")(name, param);
      }
    } else if (type == "str") {
      std::string param;
      if (nh.getParam(name, param)) {
        param_.attr("set_value")(name, param);
      }
    } else if (type == "bool") {
      bool param;
      if (nh.getParam(name, param)) {
        param_.attr("set_value")(name, param);
      }
    } else if (type == "int") {
      int param;
      if (nh.getParam(name, param)) {
        param_.attr("set_value")(name, param);
      }
    }
  }

  XmlRpc::XmlRpcValue subscribers;
  nh.getParam("subscribers", subscribers);

  py::dict sub_dict;
  for (auto& subscriber : subscribers) {
    const char* const name = subscriber.first.c_str();
    const auto& subscriber_params = subscriber.second;
    if (!sub_dict.contains(name)) {
      sub_dict[name] = py::dict();
    }
    for (auto iterat : subscriber_params) {
      const char* const key = iterat.first.c_str();
      const auto val = iterat.second;
      std::vector<std::string> arr;
      switch (val.getType()) {
        case XmlRpc::XmlRpcValue::TypeString:
          sub_dict[name][key] = static_cast<std::string>(val);
          break;
        case XmlRpc::XmlRpcValue::TypeInt:
          sub_dict[name][key] = static_cast<int>(val);
          break;
        case XmlRpc::XmlRpcValue::TypeDouble:
          sub_dict[name][key] = static_cast<double>(val);
          break;
        case XmlRpc::XmlRpcValue::TypeBoolean:
          sub_dict[name][key] = static_cast<bool>(val);
          break;
        case XmlRpc::XmlRpcValue::TypeArray:
          for (int32_t i = 0; i < val.size(); ++i) {
            auto elem = static_cast<std::string>(val[i]);
            arr.push_back(elem);
          }
          sub_dict[name][key] = arr;
          arr.clear();
          break;
        case XmlRpc::XmlRpcValue::TypeStruct:
          break;
        default:
          sub_dict[name][key] = py::cast(val);
          break;
      }
    }
  }
  param_.attr("subscriber_cfg") = sub_dict;

  // point cloud channel fusion
  if (!nh.hasParam("pointcloud_channel_fusions")) {
    ROS_WARN("No pointcloud_channel_fusions parameter found. Using default values.");
  }
  else {
    XmlRpc::XmlRpcValue pointcloud_channel_fusion;
    nh.getParam("pointcloud_channel_fusions", pointcloud_channel_fusion);

    py::dict pointcloud_channel_fusion_dict;
    for (auto& channel_fusion : pointcloud_channel_fusion) {
      const char* const name = channel_fusion.first.c_str();
      std::string fusion = static_cast<std::string>(channel_fusion.second);
      if (!pointcloud_channel_fusion_dict.contains(name)) {
        pointcloud_channel_fusion_dict[name] = fusion;
      }
    }
    ROS_INFO_STREAM("pointcloud_channel_fusion_dict: " << pointcloud_channel_fusion_dict);
    param_.attr("pointcloud_channel_fusions") = pointcloud_channel_fusion_dict;
  }

  // image channel fusion
  if (!nh.hasParam("image_channel_fusions")) {
    ROS_WARN("No image_channel_fusions parameter found. Using default values.");
  }
  else {
    XmlRpc::XmlRpcValue image_channel_fusion;
    nh.getParam("image_channel_fusions", image_channel_fusion);

    py::dict image_channel_fusion_dict;
    for (auto& channel_fusion : image_channel_fusion) {
      const char* const name = channel_fusion.first.c_str();
      std::string fusion = static_cast<std::string>(channel_fusion.second);
      if (!image_channel_fusion_dict.contains(name)) {
        image_channel_fusion_dict[name] = fusion;
      }
    }
    ROS_INFO_STREAM("image_channel_fusion_dict: " << image_channel_fusion_dict);
    param_.attr("image_channel_fusions") = image_channel_fusion_dict;
  }

  param_.attr("update")();
  resolution_ = py::cast<float>(param_.attr("get_value")("resolution"));
  map_length_ = py::cast<float>(param_.attr("get_value")("true_map_length"));
  map_n_ = py::cast<int>(param_.attr("get_value")("true_cell_n"));
}

void ElevationMappingWrapper::input(const RowMatrixXd& points, const std::vector<std::string>& channels, const RowMatrixXd& R,
                                    const Eigen::VectorXd& t, const double positionNoise, const double orientationNoise) {
  py::gil_scoped_acquire acquire;
  map_.attr("input_pointcloud")(Eigen::Ref<const RowMatrixXd>(points), channels, Eigen::Ref<const RowMatrixXd>(R),
                     Eigen::Ref<const Eigen::VectorXd>(t), positionNoise, orientationNoise);
}

void ElevationMappingWrapper::input_depth(const ColMatrixXf& image, const RowMatrixXd& R, const Eigen::VectorXd& t, 
                                          const RowMatrixXd& cameraMatrix, const double positionNoise, const double orientationNoise) {
  py::gil_scoped_acquire acquire;
  map_.attr("input_depth")(image, Eigen::Ref<const RowMatrixXd>(R), Eigen::Ref<const Eigen::VectorXd>(t),
                           Eigen::Ref<const RowMatrixXd>(cameraMatrix), positionNoise, orientationNoise);
}

void ElevationMappingWrapper::input_image(const std::vector<ColMatrixXf>& multichannel_image, const std::vector<std::string>& channels, const RowMatrixXd& R,
                                          const Eigen::VectorXd& t, const RowMatrixXd& cameraMatrix, int height, int width) {
  py::gil_scoped_acquire acquire;
  map_.attr("input_image")(multichannel_image, channels, Eigen::Ref<const RowMatrixXd>(R), Eigen::Ref<const Eigen::VectorXd>(t),
                           Eigen::Ref<const RowMatrixXd>(cameraMatrix), height, width);
}

void ElevationMappingWrapper::move_to(const Eigen::VectorXd& p, const RowMatrixXd& R) {
  py::gil_scoped_acquire acquire;
  map_.attr("move_to")(Eigen::Ref<const Eigen::VectorXd>(p), Eigen::Ref<const RowMatrixXd>(R));
}

void ElevationMappingWrapper::clear() {
  py::gil_scoped_acquire acquire;
  map_.attr("clear")();
}

bool ElevationMappingWrapper::exists_layer(const std::string& layerName) {
  py::gil_scoped_acquire acquire;
  return py::cast<bool>(map_.attr("exists_layer")(layerName));
}

void ElevationMappingWrapper::get_layer_data(const std::string& layerName, RowMatrixXf& map) {
  py::gil_scoped_acquire acquire;
  map = RowMatrixXf(map_n_, map_n_);
  map_.attr("get_map_with_name_ref")(layerName, Eigen::Ref<RowMatrixXf>(map));
}

void ElevationMappingWrapper::get_grid_map(grid_map::GridMap& gridMap, const std::vector<std::string>& requestLayerNames) {
  std::vector<std::string> basicLayerNames;
  std::vector<std::string> layerNames = requestLayerNames;
  std::vector<int> selection;
  for (const auto& layerName : layerNames) {
    if (layerName == "elevation") {
      basicLayerNames.push_back("elevation");
    }
  }

  RowMatrixXd pos(1, 3);
  py::gil_scoped_acquire acquire;
  map_.attr("get_position")(Eigen::Ref<RowMatrixXd>(pos));
  grid_map::Position position(pos(0, 0), pos(0, 1));
  grid_map::Length length(map_length_, map_length_);
  gridMap.setGeometry(length, resolution_, position);
  std::vector<Eigen::MatrixXf> maps;

  for (const auto& layerName : layerNames) {
    bool exists = map_.attr("exists_layer")(layerName).cast<bool>();
    if (exists) {
      RowMatrixXf map(map_n_, map_n_);
      map_.attr("get_map_with_name_ref")(layerName, Eigen::Ref<RowMatrixXf>(map));
      gridMap.add(layerName, map);
    }
  }
  gridMap.setBasicLayers(basicLayerNames);
}

void ElevationMappingWrapper::update_variance() {
  py::gil_scoped_acquire acquire;
  map_.attr("update_variance")();
}

void ElevationMappingWrapper::update_time() {
  py::gil_scoped_acquire acquire;
  map_.attr("update_time")();
}

}  // namespace elevation_mapping_cupy
