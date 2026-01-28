import cv2
import numpy as np
import os

# Verificar se o arquivo existe
video_path = 'LeakVid85.wmv'

if not os.path.exists(video_path):
    print(f"Arquivo {video_path} não encontrado!")
    print("Verifique o caminho do arquivo.")
    exit()

# Tentar abrir o vídeo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("""
    ERRO: Não foi possível abrir o vídeo!
    
    Possíveis soluções:
    1. Converter o vídeo para MP4: ffmpeg -i LeakVid85.wmv LeakVid85.mp4
    2. Instalar codecs adicionais
    3. Usar OpenCV com suporte a FFmpeg
    4. Tentar outro arquivo de vídeo
    """)
    
    # Tentar listar codecs disponíveis
    print("\nTentando codecs alternativos...")
    for fourcc in ['XVID', 'MJPG', 'MP4V', 'WMV1', 'WMV2']:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            print(f"Conseguiu abrir com codec: {fourcc}")
            break
else:
    print("Vídeo aberto com sucesso!")
    
    # Definir o kernel para operações morfológicas
    kernel = np.ones((3, 3), np.uint8)
    
    # Criar janelas para exibição
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Binarizado', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Limpo', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Erosão Ultimate', cv2.WINDOW_NORMAL)
    
    # Redimensionar janelas
    cv2.resizeWindow('Original', 640, 480)
    cv2.resizeWindow('Binarizado', 640, 480)
    cv2.resizeWindow('Limpo', 640, 480)
    cv2.resizeWindow('Erosão Ultimate', 640, 480)
    
    # Processar o vídeo
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Converter para escala de cinza (como no MATLAB: frameGray = rgb2gray(frameRGB))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Binarização adaptativa (como no MATLAB: imbinarize(frameGray,'adaptive','ForegroundPolarity','dark','Sensitivity',0.8))
        # Usando THRESH_BINARY_INV porque no MATLAB o 'ForegroundPolarity' é 'dark'
        frame_binary = cv2.adaptiveThreshold(frame_gray, 255, 
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
        
        # 3. Remover pequenos objetos (como no MATLAB: bwareaopen(frameBI,5))
        # Primeiro encontrar contornos
        contours, _ = cv2.findContours(frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Criar uma máscara para objetos grandes o suficiente
        mask = np.zeros_like(frame_binary)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5:  # Remover objetos com área menor que 5 pixels
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        frame_cleaned = mask
        
        # 4. Aproximação da erosão ultimate (como no MATLAB: bwulterode)
        # Primeiro: erosão iterativa
        eroded = frame_cleaned.copy()
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Aplicar erosão múltiplas vezes até que a imagem desapareça
        ultimate_erosion = np.zeros_like(eroded)
        count = 0
        
        while np.sum(eroded) > 0:
            eroded = cv2.erode(eroded, kernel_erode)
            if np.sum(eroded) > 0:
                ultimate_erosion = eroded.copy()
            count += 1
            if count > 20:  # Limite de segurança para evitar loop infinito
                break
        
        # Mostrar os frames processados
        cv2.imshow('Original', frame)
        cv2.imshow('Binarizado', frame_binary)
        cv2.imshow('Limpo', frame_cleaned)
        cv2.imshow('Erosão Ultimate', ultimate_erosion)
        
        # Sair com 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()