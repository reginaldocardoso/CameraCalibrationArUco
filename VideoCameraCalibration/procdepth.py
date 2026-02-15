import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthSubscriber(Node):
    def __init__(self):
        super().__init__('depth_subscriber')

        self.br = CvBridge()

        self.depth_subscription = self.create_subscription(
            Image,
            '/depth/image_raw',
            self.depth_callback,
            10
        )

        self.get_logger().info('Depth Subscriber inicializado')

    def depth_callback(self, msg):
        try:
            depth_image = self.br.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.process_and_display(depth_image)
        except Exception as e:
            self.get_logger().error(f'Erro no depth_callback: {str(e)}')

    def process_and_display(self, depth_image):
        try:
            h, w = depth_image.shape[:2]
            cx, cy = w // 2, h // 2

            # Regiões
            center_region = depth_image[cy-20:cy+20, cx-20:cx+20]
            left_region   = depth_image[cy-20:cy+20, 50:250]
            right_region  = depth_image[cy-20:cy+20, w-250:w-50]

            depth_center = np.nanmean(center_region)
            depth_left   = np.nanmean(left_region)
            depth_right  = np.nanmean(right_region)

            self.get_logger().info(
                f'Profundidade - Centro: {depth_center:.1f}, '
                f'Esquerda: {depth_left:.1f}, '
                f'Direita: {depth_right:.1f}'
            )


            depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = np.uint8(depth_vis)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Desenhos
            cv2.rectangle(depth_vis, (cx-20, cy-20), (cx+20, cy+20), (255, 255, 255), 2)
            cv2.rectangle(depth_vis, (50, cy-20), (250, cy+20), (255, 255, 255), 2)
            cv2.rectangle(depth_vis, (w-250, cy-20), (w-50, cy+20), (255, 255, 255), 2)

            cv2.putText(depth_vis, f"C: {depth_center:.1f}",
                        (cx-40, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(depth_vis, f"E: {depth_left:.1f}",
                        (80, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(depth_vis, f"D: {depth_right:.1f}",
                        (w-230, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            cv2.imshow("Depth Camera", depth_vis)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Erro no process_and_display: {str(e)}')

def main():
    rclpy.init()
    node = DepthSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
