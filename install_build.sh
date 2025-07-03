# https://leggedrobotics.github.io/elevation_mapping_cupy/getting_started/installation.html
# Python packages
# pip3 install -r requirements.txt

# torch install in Jetson Orin
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# wget https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl
# python3 -m pip install torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl
# rm -rf torch-1.13.0a0+d0d6b1f2.nv22.10-cp38-cp38-linux_aarch64.whl

# ROS packages
sudo apt-get install ros-foxy-pybind11-vendor ros-foxy-filters ros-foxy-nav2-msgs ros-foxy-pcl-ros
vcs import src < ros2_dependencies.repos
