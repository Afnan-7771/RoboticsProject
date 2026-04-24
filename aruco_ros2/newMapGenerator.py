import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
import math

class RotateByGyro(Node):
    def __init__(self):
        super().__init__('rotate_by_gyro')

        self.subscription = self.create_subscription(
            Float32,
            '/rotation',
            self.gyro_callback,
            10
        )
        self.publisher = self.create_publisher(Twist, '/rp3/cmd', 10)

        self.initial_yaw = None
        self.target_yaw = None
        self.reached_target = False

    def normalize_angle_rad(self, angle):
        """Normalize angle to [-π, π]"""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def gyro_callback(self, msg):
        current_yaw = self.normalize_angle_rad(msg.data)

        if self.initial_yaw is None:
            self.initial_yaw = current_yaw
            self.target_yaw = self.normalize_angle_rad(self.initial_yaw + math.radians(90))
            self.get_logger().info(f"Initial yaw: {math.degrees(self.initial_yaw):.2f}°")
            self.get_logger().info(f"Target yaw: {math.degrees(self.target_yaw):.2f}°")

        if not self.reached_target:
            yaw_error = self.normalize_angle_rad(self.target_yaw - current_yaw)

            twist = Twist()
            if abs(yaw_error) > math.radians(2.0):  # 2 degrees tolerance
                twist.angular.z = 0.3  # rad/s
                self.publisher.publish(twist)
                self.get_logger().info(
                    f"Rotating... Yaw: {math.degrees(current_yaw):.2f}° | Error: {math.degrees(yaw_error):.2f}°"
                )
            else:
                twist.angular.z = 0.0
                self.publisher.publish(twist)
                self.reached_target = True
                self.get_logger().info(f"✅ Reached target yaw: {math.degrees(current_yaw):.2f}°")

def main(args=None):
    rclpy.init(args=args)
    node = RotateByGyro()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
