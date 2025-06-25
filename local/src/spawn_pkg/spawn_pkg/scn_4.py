import rclpy
import math
import os
from rclpy.node import Node
from transforms3d import quaternions
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity

class SpawnNode(Node):
    """
    Spawn node for scenario 4.
    """
    def __init__(self):
        super().__init__("spawn_node")

        #Parameters
        self.declare_parameter("n", 2)
        self.n = self.get_parameter("n").value
        self.declare_parameter("distance", 2)
        self.distance = self.get_parameter("distance").value
        self.declare_parameter("obstacles_enable", False)
        self.obstacles_enable = self.get_parameter("obstacles_enable").value
        self.declare_parameter("obstacles", [])
        self.obstacles = self.get_parameter("obstacles").value

        # Get the file path for the robot model
        agv_sdf = os.path.join(get_package_share_directory("spawn_pkg"), "models", "agv", "model.sdf")
        obs_sdf = os.path.join(get_package_share_directory("spawn_pkg"), "models", "cylinder", "model.sdf")

        # Get the spawn_entity service
        client = self.create_client(SpawnEntity, "/spawn_entity")
        self.get_logger().info("Waiting for /spawn_entity service.")
        if not client.service_is_ready():
            client.wait_for_service()

        # AGV ids
        agv_ids = ['agv_' + str(i+1) for i in range(self.n)]

        # Generate AGV poses
        poses = {}
        x = self.distance/2
        y = [10, -10]
        orientation = [math.radians(-90), math.radians(90)]

        i = 0
        j = 0
        for agv in agv_ids:
            poses[agv] = [x, y[i], orientation[i]]
            i += 1
            if i > 1:
                i = 0
                x = -x
            j += 1
            if j == 4:
                x = x + self.distance
                j = 0
                    
        # Call spawn service
        for agv in agv_ids:
            # Set data for request
            request = SpawnEntity.Request()
            request.name = agv
            request.xml = open(agv_sdf, 'r').read()
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

        if self.obstacles_enable:
            i = 1
            for k in range(0, len(self.obstacles), 2):
                # Set data for request
                request = SpawnEntity.Request()
                request.name = 'cylinder_' + str(i)
                request.xml = open(obs_sdf, 'r').read()
                request.robot_namespace = 'cylinder_' + str(i)
                request.initial_pose.position.x = self.obstacles[k]
                request.initial_pose.position.y = self.obstacles[k+1]
                request.initial_pose.position.z = 0.0

                self.get_logger().info(f"Calling '/spawn_entity' service for obstacle {i}.")
                future = client.call_async(request)
                rclpy.spin_until_future_complete(self, future)
                if future.result() is not None:
                    print('response: %r' % future.result())
                else:
                    raise RuntimeError('exception while calling service: %r' % future.exception())
                
                i += 1

        self.get_logger().info("AGVs spawning finished. Closing node...")
        
def main(args=None):
    rclpy.init(args=args)
    node = SpawnNode()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()