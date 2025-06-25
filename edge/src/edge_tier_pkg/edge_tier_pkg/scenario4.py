from math import degrees

class Scenario4:
    """
    This scenario is used to evaluate the scalability of the proposed architecture.
    The number of AGVs can be set via launch file and must be even. 
    In Gazebo, the AGVs are placed facing each other. The supervisor will call MPC to make them swap their positions.
    """
    def __init__(self):

        self.get_logger().info(f"Loading scenario 4...")

    def start_scenario(self):
        self.get_logger().info('Starting scenario 4...')
        for agv_instance in self.agv_instances:
            x, y, theta = agv_instance.pose
            y = -1*y
            theta = degrees(theta)
            agv_instance.call_mpc('move', float(x), float(y), float(theta))

    def stop_scenario(self):
        self.run = False
        for agv_instance in self.agv_instances:
            agv_instance.call_mpc('stop', 0.0, 0.0, 0.0)