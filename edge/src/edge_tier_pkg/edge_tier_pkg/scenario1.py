class Scenario1:
    """
    In this validation scenario, the Supervisor will call the MPC from each AGV to move them diagonally.
    """
    def __init__(self):
        self.get_logger().info(f'Loading scenario 1...')
        self.positions = [(-7.5, 7.5, 135.0), (-7.5, -7.5, -135.0), (7.5, -7.5, -45.0), (7.5, 7.5, 45.0)]

    def start_scenario(self):
        self.get_logger().info('Starting scenario 1...')
        for agv_instance in self.agv_instances:
            i = int(agv_instance.agv_id[4])
            x = self.positions[i-1][0]
            y = self.positions[i-1][1]
            theta = self.positions[i-1][2]
            agv_instance.call_mpc('move', float(x), float(y), float(theta))

    def stop_scenario(self):
        pass
