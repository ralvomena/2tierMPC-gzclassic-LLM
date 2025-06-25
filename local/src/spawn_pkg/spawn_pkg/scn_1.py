import rclpy
import math
import os
from rclpy.node import Node
from transforms3d import quaternions
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity

class SpawnNode(Node):
    """
    Spawn node for scenario 1.
    """
    def __init__(self):
        super().__init__("spawn_node")

        # Get the file path for the robot model
        sdf_file_path = os.path.join(get_package_share_directory("spawn_pkg"), "models", "agv", "model.sdf")

        # Show file path
        print(f"robot_sdf={sdf_file_path}")

        # Get the spawn_entity service
        client = self.create_client(SpawnEntity, "/spawn_entity")
        self.get_logger().info("Waiting for /spawn_entity service.")
        if not client.service_is_ready():
            client.wait_for_service()

        # AGV ids
        n = 4
        agv_ids = ['agv_' + str(i+1) for i in range(n)]
        poses = {'agv_1': [7.5, -7.5, math.radians(135)], 'agv_2': [7.5, 7.5, math.radians(-135)],
                    'agv_3': [-7.5, 7.5, -0.7854], 'agv_4':[-7.5, -7.5, 0.7854]}
        
        for agv in agv_ids:
            # Set data for request
            request = SpawnEntity.Request()
            request.name = agv
            request.xml = open(sdf_file_path, 'r').read()
            request.robot_namespace = agv
            request.initial_pose.position.x = float(poses[agv][0])
            request.initial_pose.position.y = float(poses[agv][1])
            request.initial_pose.position.z = 0.0
            q = quaternions.axangle2quat([0, 0, 1], float(poses[agv][2]))
            request.initial_pose.orientation.w = q[0]
            request.initial_pose.orientation.x = q[1]
            request.initial_pose.orientation.y = q[2]
            request.initial_pose.orientation.z = q[3]

            self.get_logger().info(f"Calling `/spawn_entity` for {agv.upper()}")
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                print('response: %r' % future.result())
            else:
                raise RuntimeError('exception while calling service: %r' % future.exception())

        self.get_logger().info("AGVs spawning finished. Closing node...")
        
def main(args=None):
    rclpy.init(args=args)
    node = SpawnNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()