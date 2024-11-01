import numpy as np
import cv2
import yaml

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparando a matriz de pontos, por exemplo: (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Armazenando os pontos do objeto e pontos da imagem, de todas as imagens.
objpoints = [] # 3d pontos (REAL)
imgpoints = [] # 2d pontos (IMAGEM)

cap = cv2.VideoCapture(0)
found = 0
while(found < 10):  # AQUI 10, mas pode ser modificado para o número que desejar
    ret, img = cap.read() # Capture frame-by-frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encontrar os cantos do tabuleiro de xadrez
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # SE Encontrado: adicionar os pontos do objeto e da imagem (after refining them)
    if ret == True:
        objpoints.append(objp)   # A cada loop o objeto é o mesmo em 3D
        
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Desenhar e mostar os cantos
        img = cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        found += 1

    cv2.imshow('img', img)
    cv2.waitKey(10)

cap.release()
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Salvando os dados. è importante transformar a matrix em list
data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)

# Agora, pode-se abrir o arquivo salvo e printar as matrizes "calibration.yaml"
# Read YAML file
with open('calibration.yaml', 'r') as stream:
   dictionary = yaml.safe_load(stream)
camera_matrix = dictionary.get("camera_matrix")
dist_coeffs = dictionary.get("dist_coeff")
print(camera_matrix)
print(dist_coeffs)
  