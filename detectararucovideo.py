import numpy as np
import cv2

def aruco_detection():

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return None

    aruco_roi = None  # inicializa

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # pega o primeiro ArUco detectado
            pts = corners[0][0]

            x_min = int(pts[:, 0].min())
            x_max = int(pts[:, 0].max())
            y_min = int(pts[:, 1].min())
            y_max = int(pts[:, 1].max())

            aruco_roi = gray[y_min:y_max, x_min:x_max]

        cv2.imshow('Aruco Detection', frame)

        key = cv2.waitKey(1) & 0xFF

        # Pressione 's' para salvar e sair
        if key == ord('s') and aruco_roi is not None:
            binary = cv2.adaptiveThreshold(aruco_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            binary = aruco_pixels_to_logical(binary)
            break

        # Pressione 'q' para sair sem salvar
        if key == ord('q'):
            aruco_roi = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return binary

import numpy as np

def aruco_pixels_to_logical(binary_roi, cells_inner=4, border_cells=1):
    """
    Converte um ROI binário de um ArUco em matriz lógica (0/1).

    Parâmetros:
    - binary_roi: imagem binária (0 = preto, 255 = branco)
    - cells_inner: tamanho interno do ArUco (ex: 4 para DICT_4X4)
    - border_cells: número de células de borda (normalmente 1)

    Retorna:
    - matriz lógica (cells_inner x cells_inner)
      0 = preto
      1 = branco
    """

    total_cells = cells_inner + 2 * border_cells
    h, w = binary_roi.shape

    cell_h = h // total_cells
    cell_w = w // total_cells

    grid = np.zeros((total_cells, total_cells), dtype=int)

    for i in range(total_cells):
        for j in range(total_cells):
            cell = binary_roi[
                i * cell_h:(i + 1) * cell_h,
                j * cell_w:(j + 1) * cell_w
            ]

            # decisão por maioria
            grid[i, j] = 0 if np.mean(cell) < 127 else 1

    # remove a borda
    logical_inner = grid[
        border_cells:-border_cells,
        border_cells:-border_cells
    ]

    return logical_inner



if __name__ == "__main__":
    aruco_data = aruco_detection()

    if aruco_data is not None:
        print("Aruco detectado!")
        print("Shape:", aruco_data.shape)
        print("Matriz:\n", aruco_data)
    else:
        print("Nenhum ArUco capturado.")
