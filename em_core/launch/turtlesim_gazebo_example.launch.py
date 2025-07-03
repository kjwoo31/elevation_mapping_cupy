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

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.actions import SetParameter


def generate_launch_description():
    ld = LaunchDescription()

    # use sim time
    use_sim_time_param = SetParameter(name='use_sim_time', value=True)
    ld.add_action(use_sim_time_param)

    # Turtlebot gazebo 실행
    os.environ['TURTLEBOT3_MODEL'] = 'waffle'
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    turtlesim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([launch_file_dir, '/turtlebot3_world.launch.py']))
    ld.add_action(turtlesim_launch)

    # 2D scan to pointcloud2
    laserscan_to_pointcloud = Node(
        package='pointcloud_to_laserscan',
        executable='laserscan_to_pointcloud_node',
        name='laserscan_to_pointcloud',
        remappings=[('scan_in', '/scan'), ('cloud',  '/scan_to_cloud')],
        parameters=[{'target_frame': 'base_scan', 'transform_tolerance': 0.01}])
    ld.add_action(laserscan_to_pointcloud)

    # Elevation mapping
    em_param_dir = os.path.join(
        get_package_share_directory('em_core'),
        'config',
        'setups',
        'turtle_bot_gazebo',
        'parameters.yaml')
    em_node = Node(
        package='em_core',
        executable='elevation_mapping_node',
        name='elevation_mapping_node',
        parameters=[em_param_dir])
    ld.add_action(em_node)

    # RViz 시각화
    rviz_config_dir = os.path.join(
        get_package_share_directory('em_core'),
        'rviz2',
        'turtlesim_gazebo.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', [rviz_config_dir]])
    ld.add_action(rviz_node)

    return ld
