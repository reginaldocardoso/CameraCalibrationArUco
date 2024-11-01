'''
Welcome to the Camera Calibration Program!
  
This program:
  - Performs camera calibration using a chessboard.
'''
 
from __future__ import print_function 
import cv2
import numpy as np
import glob
  
# Dimensões do tabuleiro de xadrez
number_of_squares_X = 10 # Número de quadrados no eixo X(brancos e pretos)
number_of_squares_Y = 7  # Número de quadrados no eixo Y(brancos e pretos)
nX = number_of_squares_X - 1 # Número de cantos internos ao longo do eixo x
nY = number_of_squares_Y - 1 # Número de cantos internos ao longo do eixo y
square_size = 0.025 # Tamanho em metros do quadrado 
  
# Definir critérios de encerramento. Paramos quando uma precisão é atingida ou quando
# tivermos concluído um determinado número de iterações.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
 
# Definir coordenadas do mundo real para pontos no quadro de coordenadas 3D
# Os pontos do objeto são (0,0,0), (1,0,0), (2,0,0) ...., (5,8,0)
object_points_3D = np.zeros((nX * nY, 3), np.float32)  
  
# Essas são as coordenadas x e y                                              
object_points_3D[:,:2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2) 
 
object_points_3D = object_points_3D * square_size
 
# Armazene vetores de pontos 3D para todas as imagens do 
# tabuleiro de xadrez (quadro de coordenadas do mundo)
object_points = []
  
# Armazene vetores de pontos 2D para todas as imagens do 
# tabuleiro de xadrez (quadro de coordenadas da câmera)
image_points = []
  
# def main():   
# Obter o caminho do arquivo para imagens no diretório atual
images = glob.glob('ImagemCameraCalibration\*.jpg')

# Examine cada imagem do tabuleiro de xadrez, uma a uma
for image_file in images:
  # Carrega imagens
  image = cv2.imread(image_file)  
  # Converte imagem para escala de cinza
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
  # Encontre os cantos do tabuleiro de xadrez
  success, corners = cv2.findChessboardCorners(gray, (nY, nX), None)
  # Se os cantos forem encontrados pelo algoritmo, desenhe-os
  if success == True:
    # Anexar pontos do objeto
    object_points.append(object_points_3D)
    # Encontre mais pixels de canto exatos      
    corners_2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)       
    # Anexar pontos da imagem
    image_points.append(corners_2)
    # Desenhar cantos
    cv2.drawChessboardCorners(image, (nY, nX), corners_2, success)
  
  # Exibir a imagem. Usada para testes.
  cv2.imshow("Image", image) 
  # Exibir a janela por um curto período. Usado para testes.
  cv2.waitKey(1000) 
                                                                                                                      
'''
Execute a calibração da câmera para retornar a matriz da câmera, os coeficientes de distorção, 
os vetores de rotação e translação etc. 
'''
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points,image_points,gray.shape[::-1],None,None)

# Salvar parâmetros em um arquivo
cv_file = cv2.FileStorage('PhotoCalibration.yaml', cv2.FILE_STORAGE_WRITE)
cv_file.write('K', mtx)
cv_file.write('D', dist)
cv_file.release()

# Carregar os parâmetros do arquivo salvo
cv_file = cv2.FileStorage('calibration_chessboard.yaml', cv2.FILE_STORAGE_READ) 
mtx = cv_file.getNode('K').mat()
dst = cv_file.getNode('D').mat()
cv_file.release()

# Exibir as principais saídas de parâmetros do processo de calibração da câmera
print("Camera matrix:") 
print(mtx) 

print("\n Distortion coefficient:") 
print(dist) 

# Close all windows
cv2.destroyAllWindows() 
      
# if __name__ == '__main__':
#   print(__doc__)
#   main()