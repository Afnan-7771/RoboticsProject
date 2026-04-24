"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from aruco_interfaces.msg import ArucoMarkers
from tf_transformations import euler_from_quaternion
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


ROBOT_ID = 10
STOP_DISTANCE = 0.10  # meters
ANGLE_TOLERANCE = math.radians(5)   # must turn within ±5° before driving
ANGLE_DEAD_ZONE = math.radians(2)   # ignore very small angle error


# Map metadata (change if needed)
origin_x = 0.0
origin_y = 0.0
resolution = 0.0101  # meters per pixel


class FollowPixelPath(Node):
    def __init__(self):
        super().__init__('follow_pixel_path')


        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )


        self.subscription = self.create_subscription(
            ArucoMarkers,
            'aruco/markers/transformed',
            self.aruco_callback,
            qos_profile
        )
        self.cmd_pub = self.create_publisher(Twist, '/rp3/cmd', 10)


        self.linear_gain = 1.0
        self.angular_gain = 2.0


        self.robot_pos = None
        self.robot_ori = None
        self.path = self.load_path('/home/pi/ros2_ws/a_star_path.txt')
        self.current_index = 0


        if not self.path:
            self.get_logger().error("🚫 No waypoints loaded! Check the file path.")
        else:
            self.get_logger().info(f"📋 Loaded {len(self.path)} waypoints.")


        self.timer = self.create_timer(0.1, self.control_loop)  # 10Hz control loop


    def load_path(self, file_path):
        path = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    px, py = map(int, line.strip().split(','))
                    world_x = origin_x + px * resolution
                    world_y = origin_y + py * resolution
                    path.append((world_x, world_y))
        except Exception as e:
            self.get_logger().error(f"Failed to read path file: {e}")
        return path


    def aruco_callback(self, msg):
        for marker in msg.markers:
            if marker.id == ROBOT_ID:
                self.robot_pos = marker.pose.position
                self.robot_ori = marker.pose.orientation


    def control_loop(self):
        if self.robot_pos is None or self.robot_ori is None:
            self.get_logger().warn("⚠️ Robot marker not detected.")
            return


        if self.current_index >= len(self.path):
            self.get_logger().info("✅ All waypoints completed.")
            self.publish_stop()
            return


        # Target waypoint
        target_x, target_y = self.path[self.current_index]
        dx = target_x - self.robot_pos.x
        dy = target_y - self.robot_pos.y
        distance = math.hypot(dx, dy)


        # Get current yaw
        _, _, yaw = euler_from_quaternion([
            self.robot_ori.x,
            self.robot_ori.y,
            self.robot_ori.z,
            self.robot_ori.w
        ])
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi


        target_angle = math.atan2(dy, dx)
        angle_error = (target_angle - yaw + math.pi) % (2 * math.pi) - math.pi


        self.get_logger().info(
            f"🎯 Waypoint {self.current_index+1}/{len(self.path)}: "
            f"Target=({target_x:.2f}, {target_y:.2f}), "
            f"Distance={distance:.2f} m, "
            f"Yaw={math.degrees(yaw):.1f}°, "
            f"Error={math.degrees(angle_error):.1f}°"
        )


        twist = Twist()


        # Step 1: Check if we're close enough to stop
        if distance < STOP_DISTANCE:
            self.get_logger().info("📍 Reached waypoint. Advancing...")
            self.current_index += 1
            self.publish_stop()
            return


        # Step 2: Align heading first before driving
        if abs(angle_error) > ANGLE_TOLERANCE:
            twist.linear.x = 0.0
            twist.angular.z = max(min(self.angular_gain * angle_error, 0.5), -0.5)
            self.get_logger().info("🔁 Aligning heading...")
        else:
            if abs(angle_error) < ANGLE_DEAD_ZONE:
                angle_error = 0.0


            # Slow down as we approach the target
            if distance < 0.3:
                twist.linear.x = max(min(self.linear_gain * distance, 0.1), 0.05)
            else:
                twist.linear.x = min(self.linear_gain * distance, 0.2)


            twist.angular.z = max(min(self.angular_gain * angle_error, 0.5), -0.5)
            self.get_logger().info("🚗 Driving toward waypoint...")


        self.cmd_pub.publish(twist)
        self.get_logger().info(
            f"📤 Sent cmd: V={twist.linear.x:.2f} m/s, W={math.degrees(twist.angular.z):.1f}°/s"
        )


    def publish_stop(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    print("🟢 follow_pixel_path node starting...")
    rclpy.init(args=args)
    node = FollowPixelPath()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math
import time
import csv

CSV_PATH = '/home/pi/ros2_ws/a_star_commands.csv'  # path to your CSV file
ROBOT_CMD_TOPIC = '/rp3/cmd'
ODOM_TOPIC = '/rp3/odom'

class CSVNavigator(Node):
    def __init__(self):
        super().__init__('csv_navigator')
        self.publisher = self.create_publisher(Twist, ROBOT_CMD_TOPIC, 10)
        self.subscription = self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10)
        self.current_yaw = None

        self.linear_speed = 0.15  # meters per second
        self.angular_speed = math.radians(30)  # radians per second
        self.yaw_tolerance = math.radians(2)  # radians

        self.cmd_list = self.load_csv(CSV_PATH)

        # Wait until odometry is received
        while rclpy.ok() and self.current_yaw is None:
            rclpy.spin_once(self)

        self.execute_commands()

    def load_csv(self, path):
        commands = []
        with open(path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                commands.append((row['action'].strip().upper(), float(row['value'])))
        return commands

    def normalize_yaw(self, yaw_deg):
        return (yaw_deg + 180) % 360 - 180

    def odom_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        quaternion = (
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        )
        _, _, yaw = euler_from_quaternion(quaternion)
        self.current_yaw = self.normalize_yaw(math.degrees(yaw))

    def stop_robot(self):
        self.publisher.publish(Twist())

    def rotate_to_target(self, angle_deg):
        initial_yaw = self.current_yaw
        target_yaw = self.normalize_yaw(initial_yaw + angle_deg)
        self.get_logger().info(f"Rotating from {initial_yaw:.2f}° to {target_yaw:.2f}°")

        twist = Twist()
        direction = 1 if self.normalize_yaw(target_yaw - initial_yaw) > 0 else -1
        twist.angular.z = direction * self.angular_speed

        max_duration = abs(angle_deg) / math.degrees(self.angular_speed) * 2  # generous timeout
        start_time = time.time()

        while rclpy.ok():
            rclpy.spin_once(self)
            if self.current_yaw is None:
                continue

            yaw_error = self.normalize_yaw(target_yaw - self.current_yaw)
            self.get_logger().info(f"Current yaw: {self.current_yaw:.2f}°, Error: {yaw_error:.2f}°")

            if abs(yaw_error) < math.degrees(self.yaw_tolerance):
                break

            self.publisher.publish(twist)

            if time.time() - start_time > max_duration:
                self.get_logger().warn("Rotation timeout reached.")
                break

            time.sleep(0.01)

        self.stop_robot()

    def move_forward(self, distance_cm):
        distance_m = distance_cm / 100.0
        duration = distance_m / self.linear_speed
        self.get_logger().info(f"Moving forward {distance_cm} cm for {duration:.2f}s")

        twist = Twist()
        twist.linear.x = self.linear_speed
        self.publisher.publish(twist)
        time.sleep(duration)
        self.stop_robot()

    def execute_commands(self):
        for action, value in self.cmd_list:
            if action == 'MOVE':
                self.move_forward(value)
            elif action == 'TURN':
                self.rotate_to_target(value)
            else:
                self.get_logger().warn(f"Unknown command: {action}")
            time.sleep(0.5)  # Small delay between commands

def main(args=None):
    rclpy.init(args=args)
    node = CSVNavigator()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

