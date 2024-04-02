# Elevation Mapping cupy

[Documentation](https://leggedrobotics.github.io/elevation_mapping_cupy/)

## Quick instructions to run

### Installation

First, clone to your catkin_ws

```zsh
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone https://github.com/kjwoo31/elevation_mapping_cupy.git
```

Install following [Document](https://leggedrobotics.github.io/elevation_mapping_cupy/getting_started/installation.html)

### Build package

```zsh
cd $HOME/catkin_ws
catkin build elevation_mapping_cupy
catkin build convex_plane_decomposition_ros  # If you want to use plane segmentation
catkin build semantic_sensor  # If you want to use semantic sensors
```

### Run turtlebot example

![Elevation map examples](docs/media/turtlebot.png)

```bash
export TURTLEBOT3_MODEL=waffle
roslaunch elevation_mapping_cupy turtlesim_simple_example.launch
```

For fusing semantics into the map such as rgb from a multi modal pointcloud:

```bash
export TURTLEBOT3_MODEL=waffle
roslaunch elevation_mapping_cupy turtlesim_semantic_pointcloud_example.launch
```

For fusing semantics into the map such as rgb semantics or features from an image:

```bash
export TURTLEBOT3_MODEL=waffle
roslaunch elevation_mapping_cupy turtlesim_semantic_image_example.launch
```

For plane segmentation:

```bash
catkin build convex_plane_decomposition_ros
export TURTLEBOT3_MODEL=waffle
roslaunch elevation_mapping_cupy turtlesim_plane_decomposition_example.launch
```

To control the robot with a keyboard, a new terminal window needs to be opened.
Then run

```bash
export TURTLEBOT3_MODEL=waffle
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

Velocity inputs can be sent to the robot by pressing the keys `a`, `w`, `d`, `x`. To stop the robot completely, press `s`.

## Citing

If you use the Elevation Mapping CuPy, please cite the following paper:
Elevation Mapping for Locomotion and Navigation using GPU

[Elevation Mapping for Locomotion and Navigation using GPU](https://arxiv.org/abs/2204.12876)

Takahiro Miki, Lorenz Wellhausen, Ruben Grandia, Fabian Jenelten, Timon Homberger, Marco Hutter  

```bibtex
@inproceedings{miki2022elevation,
  title={Elevation mapping for locomotion and navigation using gpu},
  author={Miki, Takahiro and Wellhausen, Lorenz and Grandia, Ruben and Jenelten, Fabian and Homberger, Timon and Hutter, Marco},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={2273--2280},
  year={2022},
  organization={IEEE}
}
```

If you use the Multi-modal Elevation Mapping for color or semantic layers, please cite the following paper:

[MEM: Multi-Modal Elevation Mapping for Robotics and Learning](https://arxiv.org/abs/2309.16818v1)

Gian Erni, Jonas Frey, Takahiro Miki, Matias Mattamala, Marco Hutter

```bibtex
@inproceedings{erni2023mem,
  title={MEM: Multi-Modal Elevation Mapping for Robotics and Learning},
  author={Erni, Gian and Frey, Jonas and Miki, Takahiro and Mattamala, Matias and Hutter, Marco},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={11011--11018},
  year={2023},
  organization={IEEE}
}
```