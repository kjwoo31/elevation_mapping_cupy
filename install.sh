# https://leggedrobotics.github.io/elevation_mapping_cupy/getting_started/installation.html
# Python packages
pip3 install -r requirements.txt

# torch install in Jetson Orin
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
wget https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl
python3 -m pip install torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl
rm -rf torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl

# ROS packages
sudo apt-get install ros-foxy-pybind11-vendor ros-foxy-filters ros-foxy-nav2-msgs ros-foxy-pcl-ros
vcs import src < ros2_dependencies.repos


pip install cupy-cuda12x==12.2.0
install 경로 수정
git clone https://github.com/ros-perception/perception_pcl.git
cd perception_pcl/
git checkout humble
pip install "numpy>=1.17.3,<1.25.0"
export CPATH=/usr/include/python3.10:$CPATH
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
colcon build --packages-select pcl_ros em_core em_interface grid_map_msgs grid_map_ros grid_map_cmake_helpers grid_map_core grid_map_cv grid_map_rviz_plugin
source install/setup.bash
ros2 launch em_core gaemi_rosbag_example.launch.py

Warning
[elevation_mapping_node-1] /home/rc/.local/lib/python3.10/site-packages/numpy/core/getlimits.py:518: UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
[elevation_mapping_node-1]   setattr(self, word, getattr(machar, word).flat[0])
[elevation_mapping_node-1] /home/rc/.local/lib/python3.10/site-packages/numpy/core/getlimits.py:89: UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.
[elevation_mapping_node-1]   return self._float_to_str(self.smallest_subnormal)
[elevation_mapping_node-1] /home/rc/.local/lib/python3.10/site-packages/numpy/core/getlimits.py:518: UserWarning: The value of the smallest subnormal for <class 'numpy.float32'> type is zero.
[elevation_mapping_node-1]   setattr(self, word, getattr(machar, word).flat[0])
[elevation_mapping_node-1] /home/rc/.local/lib/python3.10/site-packages/numpy/core/getlimits.py:89: UserWarning: The value of the smallest subnormal for <class 'numpy.float32'> type is zero.
[elevation_mapping_node-1]   return self._float_to_str(self.smallest_subnormal)
