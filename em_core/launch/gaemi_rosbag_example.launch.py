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
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch_ros.actions import SetParameter


def generate_launch_description():
    ld = LaunchDescription()

    # use sim time
    use_sim_time_param = SetParameter(name='use_sim_time', value=True)
    ld.add_action(use_sim_time_param)

    # Elevation mapping
    em_param_dir = os.path.join(
        get_package_share_directory('em_core'),
        'config',
        'setups',
        'gaemi',
        'parameters.yaml')
    em_node = Node(
        package='em_core',
        executable='elevation_mapping_node',
        name='elevation_mapping_node',
        parameters=[em_param_dir, {'publishers.pub_list': ['elevation_map']}])
    ld.add_action(em_node)

    # Map server
    map_config_dir = os.path.join(
        get_package_share_directory('em_core'),
        'map',
        'mod_office_5th.yaml')
    start_map_saver_server_cmd = Node(
        package='gaemi_nav_map_server',
        executable='gaemi_nav_map_server',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'yaml_filename': map_config_dir},
            {'topic_name': 'map'},
            {'frame_id': 'map_current'}],
    )
    lifecycle_config_activate = ExecuteProcess(
        cmd=[
            'sh',
            '-c',
            'ros2 lifecycle set map_server configure; ros2 lifecycle set map_server activate'],
        output='screen')
    ld.add_action(start_map_saver_server_cmd)
    ld.add_action(lifecycle_config_activate)

    # RViz 시각화
    rviz_config_dir = os.path.join(
        get_package_share_directory('em_core'),
        'rviz2',
        'gaemi_rosbag.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', [rviz_config_dir]])
    ld.add_action(rviz_node)

    # rosbag 실행
    rosbag_dir = '/mnt/OT_DATASET/gaemi_rosbag'
    rosbag_play_node = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_dir],
        output='screen')
    ld.add_action(rosbag_play_node)

    return ld
