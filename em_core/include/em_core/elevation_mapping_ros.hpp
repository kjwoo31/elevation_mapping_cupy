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

#ifndef EM_CORE__ELEVATION_MAPPING_ROS_HPP_
#define EM_CORE__ELEVATION_MAPPING_ROS_HPP_

#include <Eigen/Dense>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/projection_matrix.h>
#include <iostream>
#include <mutex>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "em_interface/msg/channel_info.hpp"
#include "grid_map_msgs/msg/grid_map.hpp"
#include "grid_map_ros/grid_map_ros.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/eigen.hpp"
#include "em_core/elevation_mapping_wrapper.hpp"

namespace em_core
{

/**
 * @brief 고도 지도 ROS 연결 클래스
 */
class ElevationMappingNode : public rclcpp::Node
{
public:
  /**
   * @brief 고도 지도 ROS 연결 클래스 생성자
   * @param node_name 노드 이름
   */
  explicit ElevationMappingNode(const std::string & node_name);

  using ImageSubscriber = message_filters::Subscriber<sensor_msgs::msg::Image>;
  using ImageSubscriberPtr = std::shared_ptr<ImageSubscriber>;
  using CameraInfoSubscriber = message_filters::Subscriber<sensor_msgs::msg::CameraInfo>;
  using CameraInfoSubscriberPtr = std::shared_ptr<CameraInfoSubscriber>;
  using CameraPolicy = message_filters::sync_policies::ApproximateTime
    <sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo>;
  using CameraSync = message_filters::Synchronizer<CameraPolicy>;
  using CameraSyncPtr = std::shared_ptr<CameraSync>;
  using ChannelInfoSubscriber = message_filters::Subscriber<em_interface::msg::ChannelInfo>;
  using ChannelInfoSubscriberPtr = std::shared_ptr<ChannelInfoSubscriber>;
  using CameraChannelPolicy = message_filters::sync_policies::ApproximateTime
    <sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo, em_interface::msg::ChannelInfo>;
  using CameraChannelSync = message_filters::Synchronizer<CameraChannelPolicy>;
  using CameraChannelSyncPtr = std::shared_ptr<CameraChannelSync>;

private:
  void pcd_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud);
  void depth_image_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr & camera_info_msg);

  // Depth 이미지를 카메라 정보 없이 subscribe하는 멤버 함수
  void manual_camera_info_depth_image_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr image_msg);

  // subscribe한 이미지를 고도 지도 관리 코드로 전달하는 멤버 함수
  void input_image_of_channels(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr & camera_info_msg,
    const std::vector<std::string> & channels);

  // RGB 채널을 가진 이미지를 subscribe하는 멤버 함수
  void image_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr & camera_info_msg);

  // RGB 이외의 채널을 가진 이미지를 subscribe하는 멤버 함수
  void image_channel_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
    const sensor_msgs::msg::CameraInfo::SharedPtr & camera_info_msg,
    const em_interface::msg::ChannelInfo::ConstSharedPtr & channel_info_msg);

  // 로봇 위치에 따라 지도 위치를 보정하고 오차를 측정하는 멤버 함수
  void update_pose();

  void update_variance();
  void update_time();

  //  publish하는 지도를 주기적으로 업데이트하는 멤버 함수
  void update_grid_map();

  // 특정 인덱스의 지도 레이어를 publish하는 멤버 함수
  void publish_map_of_index(int index);

  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::ConstSharedPtr> pcd_subs_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::ConstSharedPtr>
  manual_camera_info_depth_image_subs_;
  std::vector<ImageSubscriberPtr> image_subs_;
  std::vector<CameraInfoSubscriberPtr> camera_info_subs_;
  std::vector<CameraSyncPtr> camera_syncs_;
  std::vector<CameraChannelSyncPtr> camera_channel_syncs_;
  std::vector<rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr> map_pubs_;
  rclcpp::TimerBase::SharedPtr update_variance_timer_;
  rclcpp::TimerBase::SharedPtr update_time_timer_;
  rclcpp::TimerBase::SharedPtr update_pose_timer_;
  rclcpp::TimerBase::SharedPtr update_grid_map_timer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  ElevationMappingWrapper map_;
  std::vector<double> manual_camera_info_;
  std::string map_frame_id_;
  std::string base_frame_id_;
  std::vector<std::vector<std::string>> map_layers_;
  std::vector<std::vector<std::string>> map_basic_layers_;
  std::set<std::string> map_layers_all_;
  std::vector<double> map_fps_;
  std::set<double> map_fps_unique_;
  std::vector<rclcpp::TimerBase::SharedPtr> map_timers_;
  Eigen::Vector3d low_pass_position_;
  Eigen::Vector4d low_pass_orientation_;
  // mutex와 atomic을 사용하여 multi-thread에서 값을 보호한다.
  std::mutex map_mutex_;
  std::atomic_bool is_grid_map_updated_;
  std::mutex error_mutex_;
  grid_map::GridMap grid_map_;
  double position_error_;
  double orientation_error_;
  double position_alpha_;
  double orientation_alpha_;
};

}  // namespace em_core
#endif  // EM_CORE__ELEVATION_MAPPING_ROS_HPP_
