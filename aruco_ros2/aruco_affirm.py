import rclpy
from rclpy.node import Node
from aruco_interfaces.msg import ArucoMarkers
from tf_transformations import euler_from_quaternion
import math

class ArucoAffirm(Node):
    def __init__(self):
        super().__init__('robot_navigator_node')
        
        # Setup ROS subscribers
        self.subscription = self.create_subscription(
            ArucoMarkers,
            'aruco/transformed',
            self.markers_callback,
            10
        )
        
        # Fixed target position
        self.target = {
            'x': 0.0,
            'y': 0.0
        }

    # Get aruco marker readings
    def markers_callback(self, msg):
        for marker in msg.markers:
            self.get_logger().info(f"Marker: {marker.id}")
            self.position = {
                'x': marker.pose.position.x,
                'y': marker.pose.position.y,
                'yaw': self.euler_yaw_from_quaternion(marker.pose.orientation.z, marker.pose.orientation.w),
                'test': euler_from_quaternion([
                    marker.pose.orientation.x,
                    marker.pose.orientation.y,
                    marker.pose.orientation.z,
                    marker.pose.orientation.w
                ])[2]
            }
            self.get_logger().info(f"Distance: {self.calc_distance()} (m)")
            self.get_logger().info(f"Angle: {self.calc_angle_difference()} (rad)")
            self.get_logger().info(f"Test: {self.calc_angle_difference(do_test=True)} (rad)")
            self.get_logger().info(f"------------------------------")

    # Calucalute distance between marker position and target position
    def calc_distance(self):
        dx = self.target['x'] - self.position['x']
        dy = self.target['y'] - self.position['y']
        return (dx ** 2 + dy ** 2) ** 0.5

    # Calculate angle difference between marker angle and target angle
    def calc_angle_difference(self, do_test = False):
        dx = self.target['x'] - self.position['x']
        dy = self.target['y'] - self.position['y']
        if do_test:
            ang = math.atan2(dy, dx) - self.position['test']
        else:
            ang = math.atan2(dy, dx) - self.position['yaw']
        if ang > math.pi:
            ang -= 2 * math.pi
        elif ang < -math.pi:
            ang += 2 * math.pi
        return ang

    # Convert quaternion to Euler (yaw)
    def euler_yaw_from_quaternion(self, z, w):
        siny = 2 * (w * z)
        cosy = 1 - 2 * (z * z)
        return math.atan2(siny, cosy)
    

# Main function
def main(args = None):
    rclpy.init(args = args)
    node = ArucoAffirm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Destroy the node (explicitly)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
