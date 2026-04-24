#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv
import math
import time

class CommandReplayer(Node):
    def __init__(self):
        super().__init__('command_replayer')

        # Create publisher
        self.publisher_ = self.create_publisher(Twist, '/rp3/cmd', 10)

        # Load commands
        self.commands = self.load_commands('/home/pi/ros2_ws/movement_commands.csv')
        if not self.commands:
            self.get_logger().error('No commands loaded. Exiting.')
            rclpy.shutdown()
            return

        self.command_index = 0
        self.get_logger().info(f'Loaded {len(self.commands)} commands.')
        
        # Start timer
        self.timer = self.create_timer(0.1, self.process_next_command)
        self.executing = False
        self.end_time = None

    def load_commands(self, filepath):
        commands = []
        with open(filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                action = row['action'].strip().upper()
                value = float(row['value'])
                commands.append((action, value))
        return commands

    def process_next_command(self):
        if self.executing:
            # Check if current action duration is over
            if time.time() >= self.end_time:
                self.publisher_.publish(Twist())  # stop motion
                self.executing = False
                self.command_index += 1
        else:
            # Check if all commands are done
            if self.command_index >= len(self.commands):
                self.get_logger().info('✅ Finished all commands.')
                rclpy.shutdown()
                return

            # Start executing next command
            action, value = self.commands[self.command_index]
            twist = Twist()

            if action == 'TURN':
                angular_speed = 0.5  # rad/s
                duration = abs(math.radians(value)) / angular_speed
                twist.angular.z = angular_speed if value > 0 else -angular_speed
                self.get_logger().info(f'TURN {value:.1f} degrees ({duration:.1f}s)')
            elif action == 'DRIVE':
                linear_speed = 0.2  # m/s
                duration = abs(value) / linear_speed
                twist.linear.x = linear_speed if value >= 0 else -linear_speed
                self.get_logger().info(f'DRIVE {value:.3f} meters ({duration:.1f}s)')
            else:
                self.get_logger().warn(f'Unknown action: {action}. Skipping.')
                self.command_index += 1
                return

            self.publisher_.publish(twist)
            self.executing = True
            self.end_time = time.time() + duration

def main(args=None):
    rclpy.init(args=args)
    node = CommandReplayer()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
