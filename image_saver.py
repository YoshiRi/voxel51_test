import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import qos_profile_sensor_data
import cv2
import numpy as np
import sys

class ImageSaver(Node):
    def __init__(self, skip=1):
        super().__init__('image_saver')
        self.skip = max(1, skip)
        self.received = 0
        self.saved = 0
        self.subscription = self.create_subscription(
            CompressedImage,
            '/sensing/camera/camera5/image_raw/compressed',
            self.listener_callback,
            qos_profile_sensor_data
        )
        self.get_logger().info(f'Started image_saver with skip={self.skip}')

    def listener_callback(self, msg):
        self.received += 1
        if self.received % self.skip != 0:
            return  # Skip this frame

        np_arr = np.frombuffer(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        filename = f'images_/{self.saved:04d}.jpg'
        cv2.imwrite(filename, image_np)
        self.get_logger().info(f'Saved {filename}')
        self.saved += 1

def main(args=None):
    rclpy.init(args=args)
    skip = 1
    if len(sys.argv) > 1:
        try:
            skip = int(sys.argv[1])
        except ValueError:
            print("Usage: ros2 run your_package image_saver [skip]")
            return

    node = ImageSaver(skip)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
