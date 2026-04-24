#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
import csv
import math
from aruco_interfaces.msg import ArucoMarkers
from std_msgs.msg import Float32 

class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')

        self.get_logger().info("WaypointFollower node initialized")
        self.origin_x = None
        self.origin_y = None
        self.origin_yaw = None
        self.resolution = 0.05

        self.waypoints = []
        self.current_waypoint_index = 0

        self.cmd_pub = self.create_publisher(Twist, '/rp3/cmd', 10)
        self.odom_sub = self.create_subscription(Odometry, '/rp3/odom', self.odom_callback, 10)
        self.aruco_sub = self.create_subscription(ArucoMarkers, '/aruco/markers/transformed', self.aruco_callback, 10)
        self.current_yaw = None  # New: store gyro yaw

        self.gyro_sub = self.create_subscription(
            Float32,
            '/rotation',
            self.gyro_callback,
            10
        )
        self.timer = self.create_timer(0.1, self.control_loop)
        self.current_pose = None
        self.origin_set = False

    def load_waypoints(self, path):
        waypoints = []
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                px = int(row['x'])
                py = int(row['y'])
                x, y = self.pixel_to_world(px, py)
                waypoints.append((x, y))
        return waypoints

    def pixel_to_world(self, px, py):
        px += 24
        py -= 43
        x = px * self.resolution + self.origin_x
        y = py * self.resolution + self.origin_y
        return x, y
    

    def gyro_callback(self, msg):
        """Receive yaw in radians from gyro"""
        self.current_yaw = msg.data

    def odom_callback(self, msg):
        if not self.origin_set:
            return

        # Get current position (we no longer care about yaw from quaternion)
        p = msg.pose.pose.position
        rel_x = p.x - self.origin_x
        rel_y = p.y - self.origin_y

        if self.current_yaw is None:
            return  # Wait for yaw before using pose

        self.current_pose = (rel_x, rel_y, self.current_yaw)

    def aruco_callback(self, msg):
        if self.origin_set:
            return  # already set origin

        for marker in msg.markers:
            if marker.id == 10:
                p = marker.pose.position
                # Ignore orientation from ArUco, only set position as origin
                self.origin_x = p.x
                self.origin_y = p.y
                self.origin_set = True
                self.get_logger().info(f"Origin position set from ArUco id 10: ({self.origin_x:.2f}, {self.origin_y:.2f})")

                # Load waypoints relative to this origin
                self.waypoints = self.load_waypoints("/home/pi/ros2_ws/critical_waypoints.csv")
                self.get_logger().info(f"✅ Loaded {len(self.waypoints)} waypoints relative to ArUco origin")
                break

    def control_loop(self):
        if self.current_pose is None or self.current_waypoint_index >= len(self.waypoints):
            return

        robot_x, robot_y, robot_yaw = self.current_pose
        goal_x, goal_y = self.waypoints[self.current_waypoint_index]

        dx = goal_x - robot_x
        dy = goal_y - robot_y
        distance = math.hypot(dx, dy)
        # Logging current state
        self.get_logger().info(
            f"[ROBOT] Position: ({robot_x:.2f}, {robot_y:.2f}) | Yaw: {math.degrees(robot_yaw):.1f}°"
        )
        self.get_logger().info(
            f"[GOAL ] Waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)} → "
            f"({goal_x:.2f}, {goal_y:.2f}) | Distance: {distance:.2f} m"
        )
        # Stop if close enough
        if distance < 0.1:
            self.get_logger().info(f"🎯 Reached waypoint {self.current_waypoint_index + 1}")
            self.current_waypoint_index += 1
            self.stop()
            return

        # Desired heading
        target_yaw = math.atan2(dy, dx)
        yaw_error = self.normalize_angle(target_yaw - robot_yaw)

        twist = Twist()
        if abs(yaw_error) > 0.1:
            # Rotate in place
            twist.angular.z = 0.4 * yaw_error
        else:
            # Move forward
            twist.linear.x = 0.15
            twist.angular.z = 0.2 * yaw_error  # minor correction

        self.cmd_pub.publish(twist)

    def stop(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

def main(args=None):
    print("Hellooo")
    rclpy.init(args=args)
    node = WaypointFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
