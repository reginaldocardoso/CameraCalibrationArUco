import sys
# sys.path.append('/usr/local/python/3.5')
import os
import cv2
from cv2 import aruco
import numpy as np

# Agora carregamos todos os parâmetros de calibração da câmera
calibrationFile = "calibrationFileName.xml"
calibrationParams = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ)
camera_matrix = calibrationParams.getNode("cameraMatrix").mat()
dist_coeffs = calibrationParams.getNode("distCoeffs").mat()
'''
Caso o tipo de câmera seja fisheye, dois parâmetros adicionais devem ser
carregados do arquivo de calibração.

r = calibrationParams.getNode("R").mat()
new_camera_matrix = calibrationParams.getNode("newCameraMatrix").mat()
'''
# Em seguida, duas matrizes de mapeamento pré-calculadas são chamando a função cv2.fisheye.initUndistortRectifyMap() como (supondo que as imagens a serem processadas sejam de 1080P):
image_size = (1920, 1080)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, r, new_camera_matrix, image_size, cv2.CV_16SC2)
