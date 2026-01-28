import cv2
import numpy as np

# Abrir vídeo
video_path = 'LeakVid85.wmv'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir vídeo. Converta para MP4.")
    exit()

print("Processando vídeo... Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. Converter para cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Binarização adaptativa (equivalente ao MATLAB)
    binary = cv2.adaptiveThreshold(gray, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 8)
    
    # 3. Remover pequenos objetos
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 4. Erosão ultimate simplificada
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ultimate = cv2.erode(cleaned, kernel_erode, iterations=2)
    
    # Mostrar resultados
    cv2.imshow('Original', frame)
    cv2.imshow('1. Binarizado', binary)
    cv2.imshow('2. Limpo (sem ruido)', cleaned)
    cv2.imshow('3. Erosao Ultimate', ultimate)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()