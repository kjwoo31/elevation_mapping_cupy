cmake_minimum_required(VERSION 3.0.2)
project(elevation_map_msgs)

find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  message_generation
)

add_message_files(
  FILES
  ChannelInfo.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
)
