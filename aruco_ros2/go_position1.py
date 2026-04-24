#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from aruco_interfaces.msg import ArucoMarkers
from tf_transformations import euler_from_quaternion
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

ROBOT_ID = 10
GOAL_ID = 11
STOP_DISTANCE = 0.10
CELL_SIZE = 0.15
GRID_ORIGIN_X = -4.0
GRID_ORIGIN_Y = -4.0

class GoPosition1(Node):
    def __init__(self):
        super().__init__('go_position1')

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

        self.linear_gain = 0.2
        self.angular_gain = 0.8

        self.robot_pos = None
        self.robot_ori = None
        self.goal_pos = None
        self.goal_seen = False
        self.goal_reached = False

    def world_to_grid(self, x, y):
        gx = int((x - GRID_ORIGIN_X) / CELL_SIZE)
        gy = int((y - GRID_ORIGIN_Y) / CELL_SIZE)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = (gx * CELL_SIZE) + GRID_ORIGIN_X + (CELL_SIZE / 2)
        y = (gy * CELL_SIZE) + GRID_ORIGIN_Y + (CELL_SIZE / 2)
        return x, y

    def aruco_callback(self, msg):
        self.robot_pos = None
        self.robot_ori = None
        self.goal_pos = None

        for marker in msg.markers:
            if marker.id == ROBOT_ID:
                self.get_logger().info("🤖 Detected robot marker (ID 10)")
                self.robot_pos = marker.pose.position
                self.robot_ori = marker.pose.orientation
            elif marker.id == GOAL_ID:
                self.get_logger().info("🎯 Detected goal marker (ID 15)")
                self.goal_pos = marker.pose.position

        if self.robot_pos is None or self.goal_pos is None:
            self.get_logger().warn("⚠️ One or both markers not detected.")
            return

        # Convert goal to grid center
        gx, gy = self.world_to_grid(self.goal_pos.x, self.goal_pos.y)
        target_x, target_y = self.grid_to_world(gx, gy)

        dx = target_x - self.robot_pos.x
        dy = target_y - self.robot_pos.y
        distance = math.hypot(dx, dy)

        # Get yaw
        _, _, yaw = euler_from_quaternion([
            self.robot_ori.x,
            self.robot_ori.y,
            self.robot_ori.z,
            self.robot_ori.w
        ])
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi

        # Angle to target
        target_angle = math.atan2(dy, dx)
        angle_error = target_angle - yaw
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

        self.get_logger().info(
            f"🧭 Yaw: {math.degrees(yaw):.1f}°, "
            f"Target: {math.degrees(target_angle):.1f}°, "
            f"Error: {math.degrees(angle_error):.1f}°, "
            f"Distance: {distance:.2f} m"
        )
        self.get_logger().info(
            f"TARGET POSITION "
            f"X: {marker.pose.position.x} ,"
            f"Y: {marker.pose.position.y} ,"
        )

        twist = Twist()

        # Stop if close enough
        if distance < STOP_DISTANCE:
            if not self.goal_reached:
                self.get_logger().info("✅ Goal reached — stopping.")
                self.goal_reached = True
            self.publish_stop()
            return

        # If angle error is big, turn in place
        if abs(angle_error) > math.radians(30):
            self.get_logger().info("🔁 Turning in place...")
            twist.linear.x = 0.0
            twist.angular.z = max(min(self.angular_gain * angle_error, 1.0), -1.0)
        else:
            # Turn slightly while driving
            self.get_logger().info("🚗 Driving and correcting...")
            twist.linear.x = min(self.linear_gain * distance, 0.2)
            twist.angular.z = self.angular_gain * angle_error

        self.cmd_pub.publish(twist)
        self.get_logger().info(
            f"📤 Sent cmd: V={twist.linear.x:.2f}, W={math.degrees(twist.angular.z):.1f}°/s"
        )

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

def main(args=None):
    print("🟢 go_position1 node starting...")
    rclpy.init(args=args)
    node = GoPosition1()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
