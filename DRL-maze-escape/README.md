# DRL-maze-escape


This project employs Twin Delayed Deep Deterministic Policy Gradient (TD3), a Deep Reinforcement Learning strategy, to train a mobile robot in a ROS Gazebo simulated maze. The robot learns to navigate to a goal, detected by laser readings and presented in polar coordinates, before energy depletion. The system is developed using PyTorch, tested on ROS Noetic on Ubuntu 20.04 with Python 3.8.10 and PyTorch 1.10.

## Installation
Main dependencies: 

* [ROS Noetic](http://wiki.ros.org/noetic/Installation)
* [PyTorch](https://pytorch.org/get-started/locally/)
* [Tensorboard](https://github.com/tensorflow/tensorboard)

Clone the repository:
```shell
$ cd ~
### Clone this repo
# $ git clone [https://github.com/vdsofja/Intelligent-Robotics-Assignment3-group19.git]
```
The network can be run with a standard 2D laser, but this implementation uses a simulated [3D Velodyne sensor](https://github.com/lmark1/velodyne_simulator)

Compile the workspace:
```shell
$ cd ~/DRL-maze-escape/catkin_ws
### Compile
$ catkin_make_isolated
```

Open a terminal and set up sources:
```shell
$ export ROS_HOSTNAME=localhost
$ export ROS_MASTER_URI=http://localhost:11311
$ export ROS_PORT_SIM=11311
$ export GAZEBO_RESOURCE_PATH=~/DRL-maze-escape/catkin_ws/src/multi_robot_scenario/launch
$ source ~/.bashrc
$ cd ~/DRL-maze-escape/catkin_ws
$ source devel_isolated/setup.bash
```

We set state after collision to done and raise the punishment significantly in training phase, and turn it down (collision does not lead to done state) in test phase.

Run the training:
```shell
$ cd ~/DRL-maze-escape/TD3
$ python3 train_velodyne_td3.py
```

To check the training process on tensorboard:
```shell
$ cd ~/DRL-maze-escape/TD3
$ tensorboard --logdir runs
```

To kill the training process:
```shell
$ killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3
```

Once training is completed, test the model:
```shell
$ cd ~/DRL-maze-escape/TD3
$ python3 test_velodyne_td3.py
```
