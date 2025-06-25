import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()

    n = 12
    distance = 4

    obstacles_enable = False

    obstacles = [-6.0, 0.0, 6.0, 0.0] # x, y pairs
    
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', 
            '-s', 'libgazebo_ros_factory.so'],
        output='screen')

    spawn_node = Node(
        package="spawn_pkg",
        executable="scn_4",
        name='spawn_4',
        parameters=[{"n": n}, {"distance": distance}, 
                    {"obstacles_enable": obstacles_enable}, 
                    {"obstacles": obstacles}]
    )

    ld.add_action(gazebo)
    ld.add_action(spawn_node)
    return ld