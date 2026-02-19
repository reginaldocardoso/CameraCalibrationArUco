#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import os
from datetime import datetime
import signal
import sys

class RGBDVideoRecorder(Node):
    def __init__(self):
        super().__init__('rgbd_video_recorder')
        self.bridge = CvBridge()
        
        # Configurações
        self.fps = 30  # Ajuste conforme sua câmera
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Criar diretório com timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.expanduser(f'~/ros2_videos/rgbd_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(f'📁 Salvando vídeos em: {self.output_dir}')
        
        # Inicializar variáveis
        self.rgb_writer = None
        self.depth_writer = None
        self.rgb_topic = None
        self.depth_topic = None
        self.video_initialized = False
        self.frame_size = None
        
        # Encontrar tópicos automaticamente
        self.find_camera_topics()
        
        # Configurar signal handler para Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def find_camera_topics(self):
        """Encontra automaticamente os tópicos da câmera RGB e Depth"""
        self.get_logger().info('🔍 Procurando tópicos da câmera...')
        
        # Lista de possíveis tópicos RGB
        rgb_topics_candidates = [
            '/camera/color/image_raw',
            '/camera/rgb/image_raw',
            '/camera/image_raw',
            '/rgb/image_raw',
            '/color/image_raw',
            '/camera/color/image_rect_raw',
            '/camera/rgb/image_rect_raw',
            '/camera/left/image_raw',  # Para stereo
            '/camera/right/image_raw',
        ]
        
        # Lista de possíveis tópicos Depth
        depth_topics_candidates = [
            '/camera/depth/image_raw',
            '/camera/depth/image_rect_raw',
            '/depth/image_raw',
            '/camera/aligned_depth_to_color/image_raw',
            '/camera/depth_registered/image_raw',
        ]
        
        # Verificar tópicos disponíveis
        topics = self.get_topic_names_and_types()
        available_topics = [topic for topic, _ in topics]
        
        # Encontrar RGB topic
        for candidate in rgb_topics_candidates:
            if candidate in available_topics:
                self.rgb_topic = candidate
                self.get_logger().info(f'✅ RGB tópico encontrado: {self.rgb_topic}')
                break
        
        # Encontrar Depth topic
        for candidate in depth_topics_candidates:
            if candidate in available_topics:
                self.depth_topic = candidate
                self.get_logger().info(f'✅ Depth tópico encontrado: {self.depth_topic}')
                break
        
        # Se não encontrou, tentar buscar qualquer tópico com 'image' que possa ser RGB/Depth
        if not self.rgb_topic or not self.depth_topic:
            self.get_logger().warn('⚠️  Tópicos não encontrados nas listas padrão, buscando alternativas...')
            image_topics = [t for t in available_topics if 'image' in t]
            
            for topic in image_topics:
                if 'depth' in topic.lower() and not self.depth_topic:
                    self.depth_topic = topic
                    self.get_logger().info(f'✅ Depth tópico encontrado (alternativo): {self.depth_topic}')
                elif ('color' in topic.lower() or 'rgb' in topic.lower() or 'image' in topic.lower()) and not self.rgb_topic:
                    if topic != self.depth_topic:  # Evitar usar o mesmo tópico
                        self.rgb_topic = topic
                        self.get_logger().info(f'✅ RGB tópico encontrado (alternativo): {self.rgb_topic}')
        
        # Subscrever se encontrou os tópicos
        if self.rgb_topic:
            self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        else:
            self.get_logger().error('❌ Não foi possível encontrar tópico RGB!')
        
        if self.depth_topic:
            self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        else:
            self.get_logger().error('❌ Não foi possível encontrar tópico Depth!')
    
    def init_video_writers(self, height, width):
        """Inicializa os gravadores de vídeo"""
        if not self.video_initialized:
            # Video writer para RGB
            rgb_path = os.path.join(self.output_dir, 'rgb_video.mp4')
            self.rgb_writer = cv2.VideoWriter(rgb_path, self.codec, self.fps, (width, height))
            
            # Video writer para Depth (raw - 16-bit)
            depth_path = os.path.join(self.output_dir, 'depth_raw_video.mp4')
            self.depth_writer = cv2.VideoWriter(depth_path, self.codec, self.fps, (width, height), isColor=False)
            
            # Video writer para Depth colorizado (para visualização)
            depth_colored_path = os.path.join(self.output_dir, 'depth_colored_video.mp4')
            self.depth_colored_writer = cv2.VideoWriter(depth_colored_path, self.codec, self.fps, (width, height))
            
            self.frame_size = (width, height)
            self.video_initialized = True
            
            self.get_logger().info(f'🎥 Iniciando gravação: {width}x{height} @ {self.fps}fps')
            self.get_logger().info(f'   RGB: {rgb_path}')
            self.get_logger().info(f'   Depth Raw: {depth_path}')
            self.get_logger().info(f'   Depth Colored: {depth_colored_path}')
    
    def rgb_callback(self, msg):
        """Callback para imagens RGB"""
        try:
            # Converter ROS Image para OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Inicializar video writers se necessário
            if not self.video_initialized:
                self.init_video_writers(msg.height, msg.width)
            
            # Escrever frame
            if self.rgb_writer:
                self.rgb_writer.write(cv_image)
                
        except Exception as e:
            self.get_logger().error(f'Erro no callback RGB: {e}')
    
    def depth_callback(self, msg):
        """Callback para imagens Depth"""
        try:
            # Converter ROS Image para OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            
            # Inicializar video writers se necessário
            if not self.video_initialized:
                self.init_video_writers(msg.height, msg.width)
            
            # Escrever depth raw
            if self.depth_writer:
                self.depth_writer.write(depth_image)
            
            # Criar versão colorizada para visualização
            if self.depth_colored_writer:
                # Normalizar para 8-bit
                depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_8bit = depth_normalized.astype(np.uint8)
                
                # Aplicar colormap
                depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
                
                # Escrever frame colorizado
                self.depth_colored_writer.write(depth_colored)
                
        except Exception as e:
            self.get_logger().error(f'Erro no callback Depth: {e}')
    
    def signal_handler(self, sig, frame):
        """Handler para Ctrl+C"""
        self.get_logger().info('\n🛑 Parando gravação...')
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Limpa recursos"""
        # Liberar video writers
        if self.rgb_writer:
            self.rgb_writer.release()
        if self.depth_writer:
            self.depth_writer.release()
        if hasattr(self, 'depth_colored_writer') and self.depth_colored_writer:
            self.depth_colored_writer.release()
        
        self.get_logger().info(f'💾 Vídeos salvos em: {self.output_dir}')

def main(args=None):
    rclpy.init(args=args)
    
    recorder = RGBDVideoRecorder()
    
    try:
        recorder.get_logger().info('🎬 Gravador RGB-D iniciado! Pressione Ctrl+C para parar.')
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        recorder.get_logger().info('\n🛑 Gravação interrompida pelo usuário.')
    finally:
        recorder.cleanup()
        recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()