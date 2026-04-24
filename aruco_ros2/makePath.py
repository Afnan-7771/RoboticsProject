#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import csv
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float32
from aruco_interfaces.msg import ArucoMarkers


class WaypointFollower(Node):
    def __init__(self):
        super().__init__('waypoint_follower')

        self.get_logger().info("WaypointFollower node initialized")

        # Variables para el origen y resolución del mapa
        self.origin_x = None
        self.origin_y = None
        self.resolution = 0.0101
        self.prev_distance = None

        # Lista de waypoints y control del índice actual
        self.waypoints = []
        self.current_waypoint_index = 0

        # Estado actual del robot
        self.current_pose = None  # (x, y, yaw)
        self.current_yaw = None  # Yaw obtenido desde /rotation
        self.origin_set = False

        # QoS para la subscripción a los marcadores ArUco (reliability best effort)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publicador para comandos de velocidad
        self.cmd_pub = self.create_publisher(Twist, '/rp3/cmd', 10)

        # Suscripciones
        self.odom_sub = self.create_subscription(Odometry, '/rp3/odom', self.odom_callback, 10)
        self.rotation_sub = self.create_subscription(Float32, '/rotation', self.rotation_callback, 10)
        self.aruco_sub = self.create_subscription(ArucoMarkers, '/aruco/markers/transformed', self.aruco_callback, qos_profile)
        self.create_timer(0.1, self.control_loop) 
        # Timer para el loop de control a 10Hz
        self.prev_distance = None

    def load_waypoints(self, path):
        waypoints = []
        try:
            with open(path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    px = int(row['x'])
                    py = int(row['y'])
                    self.get_logger().info(f"Original pixels: ({px}, {py})")
                    x, y = self.pixel_to_world(px, py)
                    self.get_logger().info(f"Converted to world coords: ({x:.3f}, {y:.3f})")
                    waypoints.append((x, y))
        except Exception as e:
            self.get_logger().error(f"Error loading waypoints: {e}")
        return waypoints

    def pixel_to_world(self, px, py):
        # Ajuste basado en tu función original
        px += 24
        py -= 43
        x = px * self.resolution
        y = py * self.resolution
        return x, y

    def rotation_callback(self, msg):
        self.current_yaw = self.normalize_angle(msg.data)

    def odom_callback(self, msg):
        if not self.origin_set:
            # Esperamos a tener el origen definido para calcular posición relativa
            return
        if self.current_yaw is None:
            # Esperamos a tener la orientación del robot
            return

        p = msg.pose.pose.position
        # Cálculo relativo a la posición del origen ArUco, con signo invertido
        rel_x = (p.x - self.origin_x) * (-1)
        rel_y = (p.y - self.origin_y) * (-1)

        self.current_pose = (rel_x, rel_y, self.current_yaw)

    def aruco_callback(self, msg):
        # Solo se fija el origen una vez, con marcador id 10
        if self.origin_set:
            return

        for marker in msg.markers:
            if marker.id == 10:
                p = marker.pose.position
                self.origin_x = p.x
                self.origin_y = p.y
                self.origin_set = True
                self.get_logger().info(f"Origin set from ArUco id 10: ({self.origin_x:.3f}, {self.origin_y:.3f})")

                self.waypoints = self.load_waypoints("/home/pi/ros2_ws/critical_waypoints.csv")
                if self.waypoints:
                    self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints relative to origin")
                else:
                    self.get_logger().warn("No waypoints loaded!")
                break

    def control_loop(self):
        if self.current_pose is None or self.current_waypoint_index >= len(self.waypoints):
            return

        robot_x, robot_y, robot_yaw = self.current_pose
        goal_x, goal_y = self.waypoints[self.current_waypoint_index]

        dx = goal_x - robot_x
        dy = goal_y - robot_y
        distance = math.hypot(dx, dy)

        # Check if distance is increasing
        if self.prev_distance is not None:
            if distance > self.prev_distance + 0.02:  # add small tolerance to avoid noise
                self.get_logger().warn("Distance to waypoint increasing, stopping robot.")
                self.stop()
                return

        self.prev_distance = distance

        # Existing logic
        self.get_logger().info(
            f"[ROBOT] Position: ({robot_x:.2f}, {robot_y:.2f}) | Yaw: {math.degrees(robot_yaw):.1f}°"
        )
        self.get_logger().info(
            f"[GOAL ] Waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)} → "
            f"({goal_x:.2f}, {goal_y:.2f}) | Distance: {distance:.2f} m"
        )

        if distance < 0.1:
            self.get_logger().info(f"🎯 Reached waypoint {self.current_waypoint_index + 1}")
            self.current_waypoint_index += 1
            self.prev_distance = None  # reset for next waypoint
            self.stop()
            return

        target_yaw = math.atan2(dy, dx)
        yaw_error = self.normalize_angle(target_yaw - robot_yaw)

        twist = Twist()

        if abs(yaw_error) > 0.15:
            twist.angular.z = 0.6 * abs(yaw_error)  # keep sign * math.copysign(1, yaw_error) 
            twist.linear.x = 0.0
            if abs(twist.angular.z) < 0.1:
                twist.angular.z = math.copysign(0.1, yaw_error)
        else:
            twist.linear.x = 0.12
            twist.angular.z = 0.3 * yaw_error

        max_ang_speed = 0.8
        twist.angular.z = max(-max_ang_speed, min(max_ang_speed, twist.angular.z))

        self.get_logger().info(
            f"Target yaw: {math.degrees(target_yaw):.1f}°, "
            f"Robot yaw: {math.degrees(robot_yaw):.1f}°, "
            f"Yaw error: {math.degrees(yaw_error):.1f}°"
        )
        self.get_logger().info(
            f"Yaw error (rad): {yaw_error:.3f} | linear.x: {twist.linear.x:.2f} | angular.z: {twist.angular.z:.2f}"
        )

        self.cmd_pub.publish(twist)


    def stop(self):
        self.cmd_pub.publish(Twist())

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

def main(args=None):
    rclpy.init(args=args)
    node = WaypointFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
