from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    start = 0
    end = 12 # Launch 12 AGVs
    agv_ids = ['agv_' + str(i+1) for i in range(start, end)]

    sim_name = 'scn_4'
    save_sim_data = False
    priority = 'high'
    agv_type = 1
    agv_diam = 0.75
    d_safe = 1.0
    d_safe_obs = 1.0
    high_vel = 1.0
    medium_vel = 0.75
    low_vel = 0.5
    scenario = 4
    obstacles = 0
    scn_size = [-200, -200, 200, 200] #-x, -y, +x, +y
    limit_n = 4

    mpc_planning_nodes = []

    for agv_id in agv_ids:
        mpc_planning_nodes.append(Node(
            package="edge_tier_pkg",
            executable="mpc_node",
            name=agv_id.lower() + '_mpc',
            parameters=[{"agv_id": agv_id},
                        {"priority": priority},
                        {"sim_name": sim_name},
                        {"save_sim_data": save_sim_data},
                        {"agv_type": agv_type},
                        {"agv_diam": agv_diam},
                        {"d_safe": d_safe},
                        {"d_safe_obs": d_safe_obs},
                        {"high_vel": high_vel},
                        {"medium_vel": medium_vel},
                        {"low_vel": low_vel},
                        {"scenario": scenario},
                        {"obstacles": obstacles},
                        {"scn_size": scn_size},
                        {"limit_n": limit_n}
            ]
        ))

    supervisor_node = Node(
            package="edge_tier_pkg",
            executable="supervisor",
        )
    ld.add_action(supervisor_node)

    for node in mpc_planning_nodes:
        ld.add_action(node)
    return ld
