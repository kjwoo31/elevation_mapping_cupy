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

#include <string>
#include <memory>
#include <vector>

#include "em_core/elevation_mapping_ros.hpp"

namespace em_core
{

ElevationMappingNode::ElevationMappingNode(const std::string & node_name)
: Node(node_name)
{
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // 매개변수를 불러온다.
  std::vector<std::string> pub_list;
  std::vector<std::string> sub_list;
  double update_variance_fps;
  double update_map_time_interval;
  double update_pose_fps;
  double update_grid_map_fps;
  bool use_manual_camera_info;

  this->declare_parameter("publishers.pub_list");
  this->declare_parameter("subscribers.sub_list");
  this->declare_parameter("update_variance_fps", 0.0);
  this->declare_parameter("update_map_time_interval", 0.1);
  this->declare_parameter("update_pose_fps", 10.0);
  this->declare_parameter("map_acquire_fps", 5.0);
  this->declare_parameter("use_manual_camera_info", false);
  this->declare_parameter("map_frame", "map");
  this->declare_parameter("base_frame", "base");
  this->declare_parameter("position_lowpass_alpha", 0.2);
  this->declare_parameter("orientation_lowpass_alpha", 0.2);

  pub_list = this->get_parameter("publishers.pub_list").as_string_array();
  sub_list = this->get_parameter("subscribers.sub_list").as_string_array();
  update_variance_fps = this->get_parameter("update_variance_fps").as_double();
  update_map_time_interval = this->get_parameter("update_map_time_interval").as_double();
  update_pose_fps = this->get_parameter("update_pose_fps").as_double();
  update_grid_map_fps = this->get_parameter("map_acquire_fps").as_double();
  use_manual_camera_info = this->get_parameter("use_manual_camera_info").as_bool();
  if (use_manual_camera_info) {
    this->declare_parameter("manual_camera_info");
    manual_camera_info_ = this->get_parameter("manual_camera_info").as_double_array();
  }
  map_frame_id_ = this->get_parameter("map_frame").as_string();
  base_frame_id_ = this->get_parameter("base_frame").as_string();
  position_alpha_ = this->get_parameter("position_lowpass_alpha").as_double();
  orientation_alpha_ = this->get_parameter("orientation_lowpass_alpha").as_double();

  grid_map_.setFrameId(map_frame_id_);

  // sub_list의 각 topic을 subscribe한다.
  rclcpp::SensorDataQoS sensor_qos;
  sensor_qos.keep_last(1);
  for (std::string & key : sub_list) {
    this->declare_parameter("subscribers." + key + ".data_type");
    std::string type = this->get_parameter("subscribers." + key + ".data_type").as_string();

    // type에 맞게 subscriber를 활용한다.
    if (type == "pointcloud") {  // 1. 포인트클라우드
      this->declare_parameter("subscribers." + key + ".topic_name");
      std::string pcd_topic = this->get_parameter("subscribers." + key + ".topic_name").as_string();

      rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::ConstSharedPtr subscription =
        this->create_subscription<sensor_msgs::msg::PointCloud2>(
        pcd_topic,
        sensor_qos,
        std::bind(&ElevationMappingNode::pcd_callback, this, std::placeholders::_1));
      pcd_subs_.push_back(subscription);
      RCLCPP_INFO_STREAM(this->get_logger(), "Subscribed to PointCloud2 topic: " << pcd_topic);
    } else if (type == "depth") {  // 2. Depth 이미지
      this->declare_parameter("subscribers." + key + ".topic_name");
      std::string camera_topic =
        this->get_parameter("subscribers." + key + ".topic_name").as_string();

      // gemini cliff 카메라의 경우, camera_info가 부정확하여 매개변수로 입력한다.
      if (use_manual_camera_info) {
        rclcpp::Subscription<sensor_msgs::msg::Image>::ConstSharedPtr subscription =
          this->create_subscription<sensor_msgs::msg::Image>(
          camera_topic,
          sensor_qos,
          std::bind(
            &ElevationMappingNode::manual_camera_info_depth_image_callback,
            this,
            std::placeholders::_1));
        manual_camera_info_depth_image_subs_.push_back(subscription);
        RCLCPP_INFO_STREAM(this->get_logger(), "Subscribed to Image topic: " << camera_topic);
        // message_filter를 활용하여 이미지와 camera_info의 시간을 맞추어 subscribe한다.
      } else {
        this->declare_parameter("subscribers." + key + ".camera_info_topic_name");
        std::string info_topic = this->get_parameter(
          "subscribers." + key + ".camera_info_topic_name").as_string();

        ImageSubscriberPtr image_sub = std::make_shared<ImageSubscriber>();
        CameraInfoSubscriberPtr cam_info_sub = std::make_shared<CameraInfoSubscriber>();
        image_sub->subscribe(this, camera_topic, sensor_qos.get_rmw_qos_profile());
        image_subs_.push_back(image_sub);
        cam_info_sub->subscribe(this, info_topic, sensor_qos.get_rmw_qos_profile());
        camera_info_subs_.push_back(cam_info_sub);
        CameraSyncPtr sync = std::make_shared<CameraSync>(
          CameraPolicy(10), *image_sub, *cam_info_sub);
        sync->registerCallback(&ElevationMappingNode::depth_image_callback, this);
        camera_syncs_.push_back(sync);
        RCLCPP_INFO_STREAM(
          this->get_logger(),
          "Subscribed to Image topic: " << camera_topic << ", Camera info topic: " << info_topic);
      }
    } else if (type == "image") {  // 3. RGB 혹은 Segmentation 이미지
      this->declare_parameter("subscribers." + key + ".topic_name");
      this->declare_parameter("subscribers." + key + ".camera_info_topic_name");
      std::string camera_topic =
        this->get_parameter("subscribers." + key + ".topic_name").as_string();
      std::string info_topic =
        this->get_parameter("subscribers." + key + ".camera_info_topic_name").as_string();

      ImageSubscriberPtr image_sub = std::make_shared<ImageSubscriber>();
      CameraInfoSubscriberPtr cam_info_sub = std::make_shared<CameraInfoSubscriber>();
      image_sub->subscribe(this, camera_topic, sensor_qos.get_rmw_qos_profile());
      image_subs_.push_back(image_sub);
      cam_info_sub->subscribe(this, info_topic, sensor_qos.get_rmw_qos_profile());
      camera_info_subs_.push_back(cam_info_sub);
      // message_filter를 활용하여 이미지, camera_info, 채널 정보의 시간을 맞추어 subscribe한다.
      try {
        this->declare_parameter("subscribers." + key + ".channel_info_topic_name");
        std::string channel_info_topic = this->get_parameter(
          "subscribers." + key + ".channel_info_topic_name").as_string();

        ChannelInfoSubscriberPtr channel_info_sub = std::make_shared<ChannelInfoSubscriber>();
        channel_info_sub->subscribe(this, channel_info_topic, sensor_qos.get_rmw_qos_profile());
        CameraChannelSyncPtr sync = std::make_shared<CameraChannelSync>(
          CameraChannelPolicy(10), *image_sub, *cam_info_sub, *channel_info_sub);
        sync->registerCallback(&ElevationMappingNode::image_channel_callback, this);
        camera_channel_syncs_.push_back(sync);
        RCLCPP_INFO_STREAM(
          this->get_logger(),
          "Subscribed to Image topic: " << camera_topic << ", Camera info topic: " <<
            info_topic << ", Channel info topic: " << channel_info_topic);
      }
      // 채널 정보가 없으면 "rgb"를 사용한다.
      catch (rclcpp::exceptions::ParameterNotDeclaredException & e) {
        CameraSyncPtr sync = std::make_shared<CameraSync>(
          CameraPolicy(10), *image_sub, *cam_info_sub);
        sync->registerCallback(&ElevationMappingNode::image_callback, this);
        camera_syncs_.push_back(sync);
        RCLCPP_INFO_STREAM(
          this->get_logger(),
          "Subscribed to Image topic: " << camera_topic << ", Camera info topic: " << info_topic <<
            ". Channel info topic: Not found. Using channel: rgb");
      }
    } else {
      RCLCPP_WARN_STREAM(
        this->get_logger(),
        "Subscriber data_type [" << type <<
          "] Not valid. Supported types: pointcloud, depth, image");
      continue;
    }
  }

  // elevation_mappint_wrapper.cpp와 연결한다.
  map_.initialize(this);

  // pub_list 중 가장 높은 fps의 layer들을 publish한다.
  // 1. publish할 topic과 fps를 기록한다.
  for (std::string & topic_name : pub_list) {
    std::vector<std::string> layers_list;
    std::vector<std::string> basic_layers_list;
    this->declare_parameter("publishers." + topic_name + ".layers");
    this->declare_parameter("publishers." + topic_name + ".basic_layers");
    this->declare_parameter("publishers." + topic_name + ".fps");
    std::vector<std::string> layers =
      this->get_parameter("publishers." + topic_name + ".layers").as_string_array();
    std::vector<std::string> basic_layers = this->get_parameter(
      "publishers." + topic_name + ".basic_layers").as_string_array();
    double fps = this->get_parameter("publishers." + topic_name + ".fps").as_double();
    if (fps > update_grid_map_fps) {
      RCLCPP_WARN(
        this->get_logger(),
        R"(Fps for topic %s is larger than map_acquire_fps (%f > %f).
         The topic data will be only updated at %f fps.)",
        topic_name.c_str(), fps, update_grid_map_fps, update_grid_map_fps);
    }
    for (size_t i = 0; i < layers.size(); ++i) {
      layers_list.push_back(static_cast<std::string>(layers[i]));
    }
    for (size_t i = 0; i < basic_layers.size(); ++i) {
      basic_layers_list.push_back(static_cast<std::string>(basic_layers[i]));
    }

    rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr pub =
      this->create_publisher<grid_map_msgs::msg::GridMap>(
      topic_name,
      rclcpp::QoS(rclcpp::KeepLast(1)));
    map_pubs_.push_back(pub);
    map_layers_.push_back(layers_list);
    map_basic_layers_.push_back(basic_layers_list);
    map_fps_.push_back(fps);
    map_fps_unique_.insert(fps);
  }
  // 2. pub_list 중 가장 높은 fps의 layer들을 publish한다.
  float max_fps = -1;
  for (auto fps : map_fps_unique_) {
    std::vector<int> indices;
    if (fps >= max_fps) {
      max_fps = fps;
      map_layers_all_.clear();
    }
    for (size_t i = 0; i < map_fps_.size(); ++i) {
      if (map_fps_[i] == fps) {
        indices.push_back(i);
        if (fps >= max_fps) {
          for (const auto layer : map_layers_[i]) {
            map_layers_all_.insert(layer);
          }
        }
      }
    }
    auto cb = [this, indices]() {
        for (size_t i : indices) {
          publish_map_of_index(i);
        }
      };
    int duration = 1.0 / (fps + 0.00001) * 1000;
    rclcpp::TimerBase::SharedPtr publishTimer_ =
      this->create_wall_timer(std::chrono::milliseconds(duration), cb);
    map_timers_.push_back(publishTimer_);
  }

  // fps 매개변수에 맞게 layer를 업데이트한다.
  if (update_variance_fps > 0) {
    int duration = 1.0 / (update_variance_fps + 0.00001) * 1000;
    update_variance_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(duration),
      std::bind(&ElevationMappingNode::update_variance, this));
  }
  if (update_map_time_interval > 0) {
    int duration = update_map_time_interval * 1000;
    update_time_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(duration),
      std::bind(&ElevationMappingNode::update_time, this));
  }
  if (update_pose_fps > 0) {
    int duration = 1.0 / (update_pose_fps + 0.00001) * 1000;
    update_pose_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(duration),
      std::bind(&ElevationMappingNode::update_pose, this));
  }
  if (update_grid_map_fps > 0) {
    int duration = 1.0 / (update_grid_map_fps + 0.00001) * 1000;
    update_grid_map_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(duration),
      std::bind(&ElevationMappingNode::update_grid_map, this));
  }
  RCLCPP_INFO(this->get_logger(), "Elevation mapping initialization finished");
}

void ElevationMappingNode::publish_map_of_index(int index)
{
  if (!is_grid_map_updated_) {
    return;
  }
  grid_map_msgs::msg::GridMap msg;
  std::vector<std::string> layers;

  // update_grid_map이 map_layers_all 데이터를 변경하지 않도록 lock하여 msg로 변환한다.
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    for (const auto & layer : map_layers_[index]) {
      const bool is_layer_in_all = map_layers_all_.find(layer) != map_layers_all_.end();
      if (is_layer_in_all && grid_map_.exists(layer)) {
        layers.push_back(layer);
      } else if (map_.exists_layer(layer)) {
        ElevationMappingWrapper::RowMatrixXf map_data;
        map_.get_layer_data(layer, map_data);
        grid_map_.add(layer, map_data);
        layers.push_back(layer);
      }
    }
    if (layers.empty()) {
      return;
    }

    msg = *grid_map::GridMapRosConverter::toMessage(grid_map_, layers);
  }
  msg.basic_layers = map_basic_layers_[index];
  map_pubs_[index]->publish(msg);
}

void ElevationMappingNode::pcd_callback(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud)
{
  // 채널 정보를 얻는다.
  auto fields = cloud->fields;
  std::vector<std::string> channels;
  for (size_t it = 0; it < fields.size(); it++) {
    auto & field = fields[it];
    channels.push_back(field.name);
  }

  // pybind 사용을 위해 Eigen으로 변환한다.
  auto * pcl_pc = new pcl::PCLPointCloud2;
  pcl::PCLPointCloud2ConstPtr cloudPtr(pcl_pc);
  pcl_conversions::toPCL(*cloud, *pcl_pc);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> points =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
    pcl_pc->width * pcl_pc->height,
    channels.size()
    );
  for (size_t i = 0; i < pcl_pc->width * pcl_pc->height; ++i) {
    for (size_t j = 0; j < channels.size(); ++j) {
      float temp;
      uint point_idx = i * pcl_pc->point_step + pcl_pc->fields[j].offset;
      memcpy(&temp, &pcl_pc->data[point_idx], sizeof(float));
      points(i, j) = static_cast<double>(temp);
    }
  }

  //  map frame에서 sensor 위치를 찾는다.
  geometry_msgs::msg::TransformStamped transform_tf;
  std::string sensor_frame_id = cloud->header.frame_id;
  Eigen::Isometry3d transformation_sensor_to_map;
  try {
    transform_tf = tf_buffer_->lookupTransform(
      map_frame_id_,
      sensor_frame_id,
      cloud->header.stamp,
      rclcpp::Duration::from_seconds(1.0));
    transformation_sensor_to_map = tf2::transformToEigen(transform_tf);
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    return;
  }

  // 로봇 위치 오차를 업데이트한다.
  double position_error{0.0};
  double orientation_error{0.0};
  {
    std::lock_guard<std::mutex> lock(error_mutex_);
    position_error = position_error_;
    orientation_error = orientation_error_;
  }

  // elevation_mapping_wrapper의 input_pcd 함수로 전달한다.
  map_.input_pcd(
    points,
    channels,
    transformation_sensor_to_map.rotation(),
    transformation_sensor_to_map.translation(),
    position_error,
    orientation_error);
}

void ElevationMappingNode::depth_image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::CameraInfo::SharedPtr & camera_info_msg)
{
  // pybind 사용을 위해 Eigen으로 변환한다.
  cv::Mat image = cv_bridge::toCvShare(image_msg, image_msg->encoding)->image;
  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> camera_matrix(
    &camera_info_msg->k[0]);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_img;
  cv::cv2eigen(image, eigen_img);
  if (image_msg->encoding == "16UC1") {eigen_img /= 1000.0f;}

  // map frame에서 sensor 위치를 찾는다.
  geometry_msgs::msg::TransformStamped transform_tf;
  std::string sensor_frame_id = image_msg->header.frame_id;
  Eigen::Isometry3d transformation_sensor_to_map;
  try {
    transform_tf = tf_buffer_->lookupTransform(
      map_frame_id_,
      sensor_frame_id,
      image_msg->header.stamp,
      rclcpp::Duration::from_seconds(1.0));
    transformation_sensor_to_map = tf2::transformToEigen(transform_tf);
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    return;
  }

  // 로봇 위치 오차를 업데이트한다.
  double position_error{0.0};
  double orientation_error{0.0};
  {
    std::lock_guard<std::mutex> lock(error_mutex_);
    position_error = position_error_;
    orientation_error = orientation_error_;
  }

  // elevation_mapping_wrapper의 input_depth 함수로 전달한다.
  map_.input_depth(
    eigen_img,
    transformation_sensor_to_map.rotation(),
    transformation_sensor_to_map.translation(),
    camera_matrix,
    position_error,
    orientation_error);
}

void ElevationMappingNode::manual_camera_info_depth_image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr image_msg)
{
  // pybind 사용을 위해 Eigen으로 변환한다.
  cv::Mat image = cv_bridge::toCvShare(image_msg, image_msg->encoding)->image;
  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> camera_matrix(
    &manual_camera_info_[0]);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_img;
  cv::cv2eigen(image, eigen_img);
  if (image_msg->encoding == "16UC1") {eigen_img /= 1000.0f;}

  // map frame에서 sensor 위치를 찾는다.
  geometry_msgs::msg::TransformStamped transform_tf;
  std::string sensor_frame_id = image_msg->header.frame_id;
  Eigen::Isometry3d transformation_sensor_to_map;
  try {
    transform_tf = tf_buffer_->lookupTransform(
      map_frame_id_,
      sensor_frame_id,
      image_msg->header.stamp,
      rclcpp::Duration::from_seconds(1.0));
    transformation_sensor_to_map = tf2::transformToEigen(transform_tf);
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    return;
  }

  // 로봇 위치 오차를 업데이트한다.
  double position_error{0.0};
  double orientation_error{0.0};
  {
    std::lock_guard<std::mutex> lock(error_mutex_);
    position_error = position_error_;
    orientation_error = orientation_error_;
  }

  // elevation_mapping_wrapper의 input_depth 함수로 전달한다.
  map_.input_depth(
    eigen_img,
    transformation_sensor_to_map.rotation(),
    transformation_sensor_to_map.translation(),
    camera_matrix,
    position_error,
    orientation_error);
}

void ElevationMappingNode::input_image_of_channels(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::CameraInfo::SharedPtr & camera_info_msg,
  const std::vector<std::string> & channels)
{
  // pybind 사용을 위해 Eigen으로 변환한다.
  cv::Mat image = cv_bridge::toCvShare(image_msg, image_msg->encoding)->image;
  // 이미지는 RGB/RGBA로 encoding을 변환한다.
  if (image_msg->encoding == "bgr8") {
    cv::cvtColor(image, image, CV_BGR2RGB);
  } else if (image_msg->encoding == "bgra8") {
    cv::cvtColor(image, image, CV_BGRA2RGBA);
  }
  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> camera_matrix(
    &camera_info_msg->k[0]);
  std::vector<cv::Mat> image_split;
  std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> multichannel_image;
  cv::split(image, image_split);
  for (auto img : image_split) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_img;
    cv::cv2eigen(img, eigen_img);
    multichannel_image.push_back(eigen_img);
  }

  // sensor frame에서 map frame 위치를 찾는다. (역추적)
  geometry_msgs::msg::TransformStamped transform_tf;
  std::string sensor_frame_id = image_msg->header.frame_id;
  Eigen::Isometry3d transformation_map_to_sensor;
  try {
    transform_tf = tf_buffer_->lookupTransform(
      sensor_frame_id,
      map_frame_id_,
      image_msg->header.stamp,
      rclcpp::Duration::from_seconds(1.0));
    transformation_map_to_sensor = tf2::transformToEigen(transform_tf);
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    return;
  }

  // 이미지의 채널 수와 channels 길이가 같은지 확인한다.
  unsigned int total_channels = 0;
  for (const auto & channel : channels) {
    if (channel == "rgb") {
      total_channels += 3;
    } else {
      total_channels += 1;
    }
  }
  if (total_channels != static_cast<unsigned int>(multichannel_image.size())) {
    RCLCPP_ERROR(
      this->get_logger(),
      R"(Mismatch in the size of multichannel_image (%d), channels (%d))",
      multichannel_image.size(),
      channels.size());
    RCLCPP_ERROR_STREAM(
      this->get_logger(),
      "Current Channels: " << boost::algorithm::join(channels, ", "));
    return;
  }

  // elevation_mapping_wrapper의 input_image 함수로 전달한다.
  map_.input_image(
    multichannel_image,
    channels,
    transformation_map_to_sensor.rotation(),
    transformation_map_to_sensor.translation(),
    camera_matrix,
    image.rows,
    image.cols);
}

void ElevationMappingNode::image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::CameraInfo::SharedPtr & camera_info_msg)
{
  input_image_of_channels(image_msg, camera_info_msg, {"rgb"});
}

void ElevationMappingNode::image_channel_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_msg,
  const sensor_msgs::msg::CameraInfo::SharedPtr & camera_info_msg,
  const em_interface::msg::ChannelInfo::ConstSharedPtr & channel_info_msg)
{
  input_image_of_channels(image_msg, camera_info_msg, channel_info_msg->channels);
}

void ElevationMappingNode::update_pose()
{
  // map frame에서 로봇 위치를 찾는다.
  geometry_msgs::msg::TransformStamped transform_tf;
  const auto & time_stamp = this->get_clock()->now();
  Eigen::Isometry3d transformation_base_to_map;
  try {
    transform_tf = tf_buffer_->lookupTransform(
      map_frame_id_,
      base_frame_id_,
      time_stamp,
      rclcpp::Duration::from_seconds(1.0));
    transformation_base_to_map = tf2::transformToEigen(transform_tf);
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
    return;
  }

  // 로봇 위치에 맞게 지도를 옮긴다.
  Eigen::Vector3d position(
    transform_tf.transform.translation.x,
    transform_tf.transform.translation.y,
    transform_tf.transform.translation.z);
  map_.move_to(position, transformation_base_to_map.rotation().transpose());

  // 로봇 위치 오차를 업데이트한다.
  Eigen::Vector3d position3(
    transform_tf.transform.translation.x,
    transform_tf.transform.translation.y,
    transform_tf.transform.translation.z);
  Eigen::Vector4d orientation(
    transform_tf.transform.rotation.x,
    transform_tf.transform.rotation.y,
    transform_tf.transform.rotation.z,
    transform_tf.transform.rotation.w);
  low_pass_position_ = position_alpha_ * position3 + (1 - position_alpha_) * low_pass_position_;
  low_pass_orientation_ = orientation_alpha_ * orientation + (1 - orientation_alpha_) *
    low_pass_orientation_;
  {
    std::lock_guard<std::mutex> lock(error_mutex_);
    position_error_ = (position3 - low_pass_position_).norm();
    orientation_error_ = (orientation - low_pass_orientation_).norm();
  }
}

void ElevationMappingNode::update_variance()
{
  map_.update_variance();
}

void ElevationMappingNode::update_time()
{
  map_.update_time();
}

void ElevationMappingNode::update_grid_map()
{
  std::vector<std::string> layers(map_layers_all_.begin(), map_layers_all_.end());
  std::lock_guard<std::mutex> lock(map_mutex_);
  map_.get_grid_map(grid_map_, layers);
  grid_map_.setTimestamp(this->get_clock()->now().seconds());

  is_grid_map_updated_ = true;
}

}  // namespace em_core
