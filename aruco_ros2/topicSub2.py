import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from aruco_interfaces.msg import ArucoMarkers
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from PIL import Image
from scipy.ndimage import binary_dilation

class OccupancyMapVisualizer(Node):
    def __init__(self):
        super().__init__('occupancy_map_visualizer')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription_map = self.create_subscription(
            OccupancyGrid,
            '/occupancy/map/grid',
            self.map_callback,
            qos_profile
        )

        self.subscription_markers = self.create_subscription(
            ArucoMarkers,
            '/aruco/markers/transformed',
            self.markers_callback,
            qos_profile
        )

        self.get_logger().info("📡 Waiting for map and marker messages...")

        self.received_map = False
        self.received_markers = False
        self.latest_map = None
        self.latest_markers = []

    def map_callback(self, msg: OccupancyGrid):
        if self.received_map:
            return
        self.get_logger().info("🗺️ Received occupancy grid")
        self.latest_map = msg
        self.received_map = True
        self.try_save_image()

    def markers_callback(self, msg: ArucoMarkers):
        if self.received_markers:
            return
        self.get_logger().info(f"🏷️ Received {len(msg.markers)} ArUco markers")
        self.latest_markers = msg.markers
        self.received_markers = True
        self.try_save_image()

    def try_save_image(self):
        if not (self.received_map and self.received_markers):
            return

        msg = self.latest_map
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin.position

        self.get_logger().info(f"🧭 Grid origin: ({origin.x}, {origin.y}), resolution: {resolution}, size: {width}x{height}")

        data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # === Inflate occupied zones for path planning ===
        # === Inflate occupied zones for visual/planning buffer ===
        # === Inflate occupied zones for visual/planning buffer ===
        dilation_pixels = 10  # Moderate buffer around walls
        occupied_mask = (data > 0)

        dilated_mask = binary_dilation(
            occupied_mask,
            structure=np.ones((2 * dilation_pixels + 1, 2 * dilation_pixels + 1))
        )

        # Buffer is the area added by dilation (not originally occupied)
        buffer_mask = np.logical_and(dilated_mask, ~occupied_mask)

        # Copy original data and add a soft border
        inflated_data = data.copy()
        inflated_data[buffer_mask] = 50  # Buffer value: not fully occupied, but "discouraged" for planning



        # === Build grayscale base image ===
        # === Build grayscale base image ===
        image = np.zeros((height, width), dtype=np.uint8)
        image[inflated_data == 0] = 255       # Free space = white
        image[inflated_data > 90] = 0         # Fully occupied = black
        image[np.logical_and(inflated_data > 0, inflated_data <= 90)] = 100  # Buffer = dark gray
        image[inflated_data < 0] = 127        # Unknown = gray

        # Convert to RGB for marker drawing
        image_rgb = np.stack([image] * 3, axis=2)

        # === World to pixel conversion ===
        def world_to_pixel(x, y):
            px = int((x - origin.x) / resolution)
            py = int((y - origin.y) / resolution)
            px -= 24  # x correction
            py += 43  # y correction
            return px, py

        # === Draw markers ===
        for marker in self.latest_markers:
            marker_id = marker.id
            pos = marker.pose.position
            px, py = world_to_pixel(pos.x, pos.y)

            self.get_logger().info(f"📍 Marker {marker_id} at world ({pos.x:.2f}, {pos.y:.2f}) → pixel ({px}, {py})")

            if 0 <= px < width and 0 <= py < height:
                color = (255, 255, 0) if marker_id == 15 else \
                        (0, 0, 255) if marker_id == 10 else \
                        (0, 0, 0)

                marker_radius = 4
                for dx in range(-marker_radius, marker_radius + 1):
                    for dy in range(-marker_radius, marker_radius + 1):
                        x_pix = px + dx
                        y_pix = py + dy
                        if 0 <= x_pix < width and 0 <= y_pix < height:
                            image_rgb[y_pix, x_pix] = color

        # === Save final image ===
        save_path = "/home/pi/ros2_ws/occupancy_map_with_markers2.png"
        Image.fromarray(image_rgb).save(save_path)
        self.get_logger().info(f"✅ Saved dilated map with markers to: {save_path}")

        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyMapVisualizer()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
