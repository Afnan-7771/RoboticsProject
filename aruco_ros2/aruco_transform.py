import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from std_msgs.msg import Int64MultiArray
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from aruco_interfaces.msg import ArucoMarker, ArucoMarkers


class ArucoTransformer(Node):
    def __init__(self):
        super().__init__('aruco_tranformer_node')

        # Declare parameters
        self.declare_parameter('scalar_x', 1.0)
        self.declare_parameter('scalar_y', 1.0)
        self.declare_parameter('scalar_z', 1.0)
        self.declare_parameter('ref_marker_id', 0)
        
        # Setup ROS subscriber
        self.subscription = self.create_subscription(
            ArucoMarkers,
            'aruco/markers',
            self.marker_callback,
            10
        )
        self.publisher = self.create_publisher(
            ArucoMarkers,
            'aruco/transformed',
            10
        )

    def marker_callback(self, msg):
        ref_marker_index = -1
        for i in range(len(msg.markers)):
            if self.get_parameter("ref_marker_id").get_parameter_value().integer_value == msg.markers[i].id:
                ref_marker_index = i
                break
        if ref_marker_index < 0:
            self.get_logger().warn("Reference marker not found!")
            return
            
        # Get the pose of the reference marker
        ref_pose = msg.markers[ref_marker_index].pose

        # Create a new ArucoMarkers to store the transformed poses
        markers = ArucoMarkers()
        markers.header = msg.header
        markers.markers = []

        # Transform the poses of the other markers relative to the reference marker
        for i in range(len(msg.markers)):
            if i != ref_marker_index:
                
                # Compute the relative pose
                relative_pose = Pose()
                relative_pose.position.x = msg.markers[i].pose.position.x - ref_pose.position.x
                relative_pose.position.y = msg.markers[i].pose.position.y - ref_pose.position.y
                relative_pose.position.z = msg.markers[i].pose.position.z - ref_pose.position.z

                # Compute the relative orientation quaternion
                relative_pose.orientation = self.compute_relative_orientation(ref_pose.orientation, msg.markers[i].pose.orientation)

                # Apply scalar offset
                relative_pose.position.x = relative_pose.position.x * self.get_parameter("scalar_x").get_parameter_value().double_value
                relative_pose.position.y = relative_pose.position.y * self.get_parameter("scalar_y").get_parameter_value().double_value
                relative_pose.position.z = relative_pose.position.z * self.get_parameter("scalar_z").get_parameter_value().double_value
                
                # Add the marker id and transformed pose to the new message
                marker = ArucoMarker()
                marker.id = msg.markers[i].id
                marker.pose = relative_pose
                markers.markers.append(marker)

        # Publish the transformed poses
        self.publisher.publish(markers)
                            
    def compute_relative_orientation(self, ref_orientation, orientation):
        # Convert the reference and target orientations to Euler angles
        ref_euler = self.quaternion_to_euler(ref_orientation)
        target_euler = self.quaternion_to_euler(orientation)

        # Compute the relative Euler angles
        relative_euler = [target - ref for target, ref in zip(target_euler, ref_euler)]

        # Convert the relative Euler angles back to a quaternion
        relative_orientation = self.euler_to_quaternion(relative_euler)

        return relative_orientation

    @staticmethod
    def quaternion_to_euler(quaternion):
        # Convert a quaternion to Euler angles
        import math

        x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

    @staticmethod
    def euler_to_quaternion(euler):
        # Convert Euler angles to a quaternion
        import math
        import tf2_ros
        from geometry_msgs.msg import Quaternion

        roll, pitch, yaw = euler

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        quaternion = Quaternion()
        quaternion.w = cy * cp * cr + sy * sp * sr
        quaternion.x = cy * cp * sr - sy * sp * cr
        quaternion.y = sy * cp * sr + cy * sp * cr
        quaternion.z = sy * cp * cr - cy * sp * sr

        return quaternion


def main(args=None):
    rclpy.init(args = args)
    node = ArucoTransformer()
    rclpy.spin(node)
    marker_listener.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

