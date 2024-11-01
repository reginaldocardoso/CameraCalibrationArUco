'''
Bem-vindo ao ArUco Marker estimador de posição!

pip uninstall opencv-python -y
pip uninstall opencv-contrib-python -y
pip install opencv-contrib-python==4.6.0.66
  
Este Programa:
  - Estima a posição de um único ArUco Marker
'''
  
from __future__ import print_function
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import os

if not os.path.exists(os.path.join(os.getcwd(), 'Dados')):
    os.makedirs(os.path.join(os.getcwd(), 'Dados'))

Roll_x = []
Pitch_y = []
Yaw_z = []

Pos_x = []
Pos_y = []
Pos_z = []

def euler_from_quaternion(x, y, z, w):
  """
    Converter quaternion para ângulos de Euler (roll, pitch, yaw)
    roll é a rotação em torno de x em radianos (sentido anti-horário)
    pitch é a rotação em torno de y em radianos (sentido anti-horário)
    yaw é a rotação em torno de z em radianos (sentido anti-horário)
  """
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll_x = math.atan2(t0, t1)
      
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch_y = math.asin(t2)
      
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z = math.atan2(t3, t4)
      
  return roll_x, pitch_y, yaw_z # radians

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
  '''
  Isso estimará o rvec e o tvec para cada um dos cantos dos marcadores detectados por:
     corners, ids, rejectedImgPoints = detector.detectMarkers(image)
  corners - é uma matriz de cantos detectados para cada marcador detectado na imagem
  marker_size - é o tamanho dos marcadores detectados
  mtx - é a matriz da câmera
  distortion - é a matriz de distorção da câmera
  RETURN lista de rvecs, tvecs e trash (de modo que corresponda ao antigo estimatePoseSingleMarkers())
  '''
  marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                            [marker_size / 2, marker_size / 2, 0],
                            [marker_size / 2, -marker_size / 2, 0],
                            [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
  trash = []
  rvecs = []
  tvecs = []
    
  for c in corners:
    nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
    rvecs.append(R)
    tvecs.append(t)
    trash.append(nada)
  return rvecs, tvecs, trash


# Dicionário que foi usado para gerar o marcador ArUco
aruco_dictionary_name = "DICT_5X5_100"
 
# Os diferentes dicionários ArUco incorporados à biblioteca OpenCV. 
ARUCO_DICT = {
  "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
  "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
  "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
  "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
  "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
  "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
  "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
  "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
  "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
  "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
  "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
  "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
  "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
  "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
  "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
  "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
  "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}
 
# Comprimento lateral do marcador ArUco em metros 
aruco_marker_side_length = 0.06
 
# Arquivo yaml de parâmetros de calibração
camera_calibration_parameters_filename = 'PhotoCalibration.yaml'
 
# Verificar se temos um marcador ArUco válido
# if ARUCO_DICT.get(aruco_dictionary_name, None) is None:
#   print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
#   sys.exit(0)
 
# Carregar os parâmetros da câmera a partir do arquivo salvo
cv_file = cv2.FileStorage(
    camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
mtx = cv_file.getNode('K').mat()
dst = cv_file.getNode('D').mat()
cv_file.release()
     
# Carregar o dicionário ArUco
print("[INFO] detecting '{}' markers...".format(
    aruco_dictionary_name))
this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_name])
this_aruco_parameters = cv2.aruco.DetectorParameters()
   
# Iniciar Video
cap = cv2.VideoCapture(0)
while(True):    
  # Captura quadro a quadro
  # Esse método retorna True/False, assim como
  # como o quadro de vídeo.
  ret, frame = cap.read()
  imgRemapped_gray = frame
   
  # Detectar marcadores ArUco no quadro de vídeo
  detector = cv2.aruco.ArucoDetector(this_aruco_dictionary, this_aruco_parameters)#,cameraMatrix=mtx, distCoeff=dst)
  (corners, marker_ids, rejected) = detector.detectMarkers(imgRemapped_gray)

  # Verificar se pelo menos um marcador ArUco foi detectado
  if marker_ids is not None:
    # Desenhe um quadrado ao redor dos marcadores detectados no quadro de vídeo
    cv2.aruco.drawDetectedMarkers(imgRemapped_gray, corners, marker_ids)
    # Obter os vetores de rotação e translação  
    rvecs, tvecs, trash = my_estimatePoseSingleMarkers(corners,aruco_marker_side_length,mtx,dst)       
         
    # Imprimir a pose do marcador ArUco
    # A pose do marcador é em relação à moldura da lente da câmera.
    # Imagine que você está olhando pelo visor da câmera, 
    # O quadro da lente da câmera:
    # O eixo x aponta para a direita
    # O eixo y aponta diretamente para baixo, em direção aos dedos dos pés
    # O eixo z aponta para a frente, longe do seu olho, para fora da câmera
    for i, marker_id in enumerate(marker_ids):
      ####################################################################
      # BATATA
      ###################################################################
      # Armazenar as informações de conversão (ou seja, posição)
      # print(tvecs)
      # print(tvecs.__sizeof__())      
      transform_translation_x = tvecs[i][0]#[0]
      transform_translation_y = tvecs[i][1]#[1]
      transform_translation_z = tvecs[i][2]#[2]
 
      # Armazenar as informações de rotação
      rotation_matrix = np.eye(4)
      rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs))[0]
      r = R.from_matrix(rotation_matrix[0:3, 0:3])
      quat = r.as_quat()   
         
      # Quaternion formato     
      transform_rotation_x = quat[0] 
      transform_rotation_y = quat[1] 
      transform_rotation_z = quat[2] 
      transform_rotation_w = quat[3] 
       
      # Euler angle formato em radians
      roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
                                                       transform_rotation_y, 
                                                       transform_rotation_z, 
                                                       transform_rotation_w)
         
      roll_x = math.degrees(roll_x)
      pitch_y = math.degrees(pitch_y)
      yaw_z = math.degrees(yaw_z)
      print("transform_translation_x: {}".format(transform_translation_x))
      print("transform_translation_y: {}".format(transform_translation_y))
      print("transform_translation_z: {}".format(transform_translation_z))
      print("roll_x: {}".format(roll_x))
      print("pitch_y: {}".format(pitch_y))
      print("yaw_z: {}".format(yaw_z))
      print()
      
      Roll_x.append(roll_x)
      Pitch_y.append(pitch_y)
      Yaw_z.append(yaw_z)   
      Pos_x.append(transform_translation_x)
      Pos_y.append(transform_translation_y)
      Pos_z.append(transform_translation_z) 
      # Desenhe os eixos no marcador
      cv2.drawFrameAxes(imgRemapped_gray, mtx, dst, rvecs[i], tvecs[i], 0.05, 3)
     
  # Exibir o quadro resultante
  cv2.imshow('frame',imgRemapped_gray)

  np.savetxt('dados/PosX.txt', Pos_x)
  np.savetxt('dados/PosY.txt', Pos_y)
  np.savetxt('dados/PosZ.txt', Pos_z)
  np.savetxt('dados/AttR.txt', Roll_x)
  np.savetxt('dados/AttP.txt', Pitch_y)
  np.savetxt('dados/AttY.txt', Yaw_z)
          
  # Se “q” for pressionado no teclado, sairá desse loop
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()

