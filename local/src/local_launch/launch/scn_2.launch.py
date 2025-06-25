import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()

    world_file_name = 'scenario2.world'
    pkg_dir = get_package_share_directory('spawn_pkg')

    os.environ["GAZEBO_MODEL_PATH"] = os.path.join(pkg_dir, 'models')

    world = os.path.join(pkg_dir, 'worlds', world_file_name)

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_init.so', 
            '-s', 'libgazebo_ros_factory.so'],
        output='screen')

    spawn_node = Node(
        package="spawn_pkg",
        executable="scn_2",
        name='spawn_scn_2',
    )

    # netem_gui_node = Node(
    #     package="local_tier_pkg",
    #     executable="netem_gui_node",
    # )

    ld.add_action(gazebo)
    ld.add_action(spawn_node)
    # ld.add_action(netem_gui_node)
    return ld