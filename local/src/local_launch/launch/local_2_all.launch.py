from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    agvs_quantity = 8
    agv_ids = ['agv_' + str(i+1) for i in range(agvs_quantity)]
    mpc_local_nodes = []

    priority = 'high'
    sim_name = 'scn_2'
    save_sim_data = False
    high_vel = 1.0
    medium_vel = 0.75
    low_vel = 0.5
    scenario = 2
    scn_size = [-25, -25, 25, 25] #-x, -y, +x, +y

    for agv_id in agv_ids:
        mpc_local_nodes.append(Node(
            package="local_tier_pkg",
            executable="mpc_tracking_node",
            name=agv_id.lower() + '_mpc_local_tracking',
            parameters=[{"agv_id": agv_id},
                    {"priority": priority},
                    {"sim_name": sim_name},
                    {"save_sim_data": save_sim_data},
                    {"high_vel": high_vel},
                    {"medium_vel": medium_vel},
                    {"low_vel": low_vel},
                    {"scenario": scenario},
                    {"scn_size": scn_size}
        ]
        ))

    for node in mpc_local_nodes:
        ld.add_action(node)
    return ld