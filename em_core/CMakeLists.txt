cmake_minimum_required(VERSION 3.8)
project(em_core)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(em_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pybind11 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

set(DEPENDENCIES
  em_interface
  geometry_msgs
  grid_map_msgs
  grid_map_ros
  pcl_ros
  pybind11
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  tf2
  tf2_eigen
)

set(DETECTOR_LIB_SRC
  src/elevation_mapping_ros.cpp
  src/elevation_mapping_wrapper.cpp
)

add_library(em_core ${DETECTOR_LIB_SRC})

target_include_directories(em_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(em_core ${DEPENDENCIES})

add_executable(elevation_mapping_node src/elevation_mapping_node.cpp)

target_link_libraries(elevation_mapping_node em_core pybind11::embed)

install(TARGETS elevation_mapping_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config launch map rosbag rviz2
  DESTINATION share/${PROJECT_NAME})

ament_python_install_package(${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
