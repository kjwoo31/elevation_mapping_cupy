//
// Copyright (c) 2022, Takahiro Miki. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#pragma once

// STL
#include <iostream>
#include <mutex>

// Eigen
#include <Eigen/Dense>

// Pybind
#include <pybind11/embed.h>  // everything needed for embedding

// ROS
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>

// Grid Map
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>

// PCL
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <elevation_map_msgs/ChannelInfo.h>

#include "elevation_mapping_cupy/elevation_mapping_wrapper.hpp"

namespace py = pybind11;

namespace elevation_mapping_cupy {

class ElevationMappingNode {
 public:
  ElevationMappingNode(ros::NodeHandle& nh);
  using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ColMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

  using ImageSubscriber = image_transport::SubscriberFilter;
  using ImageSubscriberPtr = std::shared_ptr<ImageSubscriber>;

  // Subscriber and Synchronizer for CameraInfo messages
  using CameraInfoSubscriber = message_filters::Subscriber<sensor_msgs::CameraInfo>;
  using CameraInfoSubscriberPtr = std::shared_ptr<CameraInfoSubscriber>;
  using CameraPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo>;
  using CameraSync = message_filters::Synchronizer<CameraPolicy>;
  using CameraSyncPtr = std::shared_ptr<CameraSync>;

  // Subscriber and Synchronizer for ChannelInfo messages
  using ChannelInfoSubscriber = message_filters::Subscriber<elevation_map_msgs::ChannelInfo>;
  using ChannelInfoSubscriberPtr = std::shared_ptr<ChannelInfoSubscriber>;
  using CameraChannelPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, elevation_map_msgs::ChannelInfo>;
  using CameraChannelSync = message_filters::Synchronizer<CameraChannelPolicy>;
  using CameraChannelSyncPtr = std::shared_ptr<CameraChannelSync>;

  // Subscriber and Synchronizer for Pointcloud messages
  using PointCloudSubscriber = message_filters::Subscriber<sensor_msgs::PointCloud2>;
  using PointCloudSubscriberPtr = std::shared_ptr<PointCloudSubscriber>;
  using PointCloudPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, elevation_map_msgs::ChannelInfo>;
  using PointCloudSync = message_filters::Synchronizer<PointCloudPolicy>;
  using PointCloudSyncPtr = std::shared_ptr<PointCloudSync>;

 private:
  void readParameters();
  void setupMapPublishers();
  void pointcloudCallback(const sensor_msgs::PointCloud2& cloud, const std::string& key);
  void inputPointCloud(const sensor_msgs::PointCloud2& cloud, const std::vector<std::string>& channels);
  void depthCallback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& camera_info_msg);
  void inputImage(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& camera_info_msg, const std::vector<std::string>& channels);
  void imageCallback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& camera_info_msg, const std::string& key);
  void imageChannelCallback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& camera_info_msg, const elevation_map_msgs::ChannelInfoConstPtr& channel_info_msg);
  void pointCloudChannelCallback(const sensor_msgs::PointCloud2& cloud, const elevation_map_msgs::ChannelInfoConstPtr& channel_info_msg);
  void updatePose(const ros::TimerEvent&);
  void updateVariance(const ros::TimerEvent&);
  void updateTime(const ros::TimerEvent&);
  void updateGridMap(const ros::TimerEvent&);
  void publishMapToOdom(double error);
  void publishMapOfIndex(int index);

  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  std::vector<ros::Subscriber> pointcloudSubs_;
  std::vector<ImageSubscriberPtr> imageSubs_;
  std::vector<CameraInfoSubscriberPtr> cameraInfoSubs_;
  std::vector<ChannelInfoSubscriberPtr> channelInfoSubs_;
  std::vector<CameraSyncPtr> cameraSyncs_;
  std::vector<CameraChannelSyncPtr> cameraChannelSyncs_;
  std::vector<PointCloudSyncPtr> pointCloudSyncs_;
  std::vector<ros::Publisher> mapPubs_;
  ros::Timer updateVarianceTimer_;
  ros::Timer updateTimeTimer_;
  ros::Timer updatePoseTimer_;
  ros::Timer updateGridMapTimer_;
  tf::TransformListener transformListener_;
  ElevationMappingWrapper map_;
  std::string mapFrameId_;
  std::string baseFrameId_;

  // map topics info
  std::vector<std::vector<std::string>> map_layers_;
  std::vector<std::vector<std::string>> map_basic_layers_;
  std::set<std::string> map_layers_all_;
  std::vector<double> map_fps_;
  std::set<double> map_fps_unique_;
  std::vector<ros::Timer> mapTimers_;
  std::map<std::string, std::vector<std::string>> channels_;

  Eigen::Vector3d lowpassPosition_;
  Eigen::Vector4d lowpassOrientation_;

  std::mutex mapMutex_;  // protects gridMap_
  grid_map::GridMap gridMap_;
  std::atomic_bool isGridmapUpdated_;  // needs to be atomic (read is not protected by mapMutex_)

  std::mutex errorMutex_;  // protects positionError_, and orientationError_
  double positionError_;
  double orientationError_;

  double positionAlpha_;
  double orientationAlpha_;
};

}  // namespace elevation_mapping_cupy
