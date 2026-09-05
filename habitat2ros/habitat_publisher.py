import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Quaternion, Point, PoseStamped
from habitat.core.simulator import Observations
import numpy as np
from copy import deepcopy


def quaternion_from_euler(roll, pitch, yaw):
    """Return an ROS-order (x, y, z, w) quaternion without the tf2 ABI."""
    roll = float(np.asarray(roll).reshape(-1)[0])
    pitch = float(np.asarray(pitch).reshape(-1)[0])
    yaw = float(np.asarray(yaw).reshape(-1)[0])

    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5
    sr, cr = np.sin(half_roll), np.cos(half_roll)
    sp, cp = np.sin(half_pitch), np.cos(half_pitch)
    sy, cy = np.sin(half_yaw), np.cos(half_yaw)

    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class ROSPublisher:
    def __init__(self):
        # Create ROS publishers
        self.depth_pub = rospy.Publisher("/habitat/camera_depth", Image, queue_size=10)
        self.rgb_pub = rospy.Publisher("/habitat/camera_rgb", Image, queue_size=10)
        self.odom_pub = rospy.Publisher("/habitat/odom", Odometry, queue_size=10)
        self.pose_pub = rospy.Publisher("/habitat/sensor_pose", Odometry, queue_size=10)
    def publish_depth(self, ros_time, depth_image):
        depth_image = np.asarray(depth_image, dtype=np.float32)
        if depth_image.ndim == 3 and depth_image.shape[-1] == 1:
            depth_image = depth_image[..., 0]
        if depth_image.ndim != 2:
            raise ValueError(f"Expected HxW depth image, got {depth_image.shape}")
        depth_image = np.ascontiguousarray(depth_image)
        depth_msg = Image()
        depth_msg.height, depth_msg.width = depth_image.shape
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = False
        depth_msg.step = depth_msg.width * depth_image.dtype.itemsize
        depth_msg.data = depth_image.tobytes()
        depth_msg.header.stamp = ros_time
        depth_msg.header.frame_id = "world"
        self.depth_pub.publish(depth_msg)

    def publish_rgb(self, ros_time, rgb_image):
        rgb_image = np.ascontiguousarray(np.asarray(rgb_image, dtype=np.uint8))
        rgb_msg = Image()
        rgb_msg.height, rgb_msg.width = rgb_image.shape[:2]
        rgb_msg.encoding = "rgb8"
        rgb_msg.is_bigendian = False
        rgb_msg.step = rgb_msg.width * 3
        rgb_msg.data = rgb_image.tobytes()
        rgb_msg.header.stamp = ros_time
        rgb_msg.header.frame_id = "world"
        self.rgb_pub.publish(rgb_msg)

    def publish_robot_odom(self, ros_time, gps, compass):
        copy_compass = deepcopy(compass)
        odom = Odometry()
        odom.header.stamp = ros_time
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose = Pose(
            position=Point(-gps[2], -gps[0], gps[1]),
            orientation=Quaternion(*quaternion_from_euler(0, 0, copy_compass)),
        )
        self.odom_pub.publish(odom)

    def publish_camera_odom(self, ros_time, gps, compass, pitch):
        copy_compass = deepcopy(compass)
        copy_pitch = deepcopy(pitch)
        sensor_pose = Odometry()
        sensor_pose.header.stamp = ros_time
        sensor_pose.header.frame_id = "world"
        sensor_pose.child_frame_id = "base_link"
        sensor_pose.pose.pose = Pose(
            position=Point(-gps[2], -gps[0], gps[1] + 0.88),
            orientation=Quaternion(
                *quaternion_from_euler(
                    copy_pitch + np.pi / 2.0, np.pi, copy_compass + np.pi / 2.0
                )
            ),
        )
        self.pose_pub.publish(sensor_pose)

    def habitat_publish_ros_topic(self, observations):
        depth_image = observations["depth"]
        rgb_image = observations["rgb"]
        gps = observations["gps"]
        compass = observations["compass"]
        camera_pitch = observations["camera_pitch"]
        ros_time = rospy.Time.now()
        self.publish_depth(ros_time, depth_image)
        self.publish_camera_odom(ros_time, gps, compass, camera_pitch)
        self.publish_rgb(ros_time, rgb_image)
        self.publish_robot_odom(ros_time, gps, compass)
