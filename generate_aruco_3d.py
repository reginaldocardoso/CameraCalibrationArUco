# generate_aruco_3d.py
import numpy as np
from stl import mesh
import cv2

def generate_3d_aruco_marker(marker_id, size_cm=15, relief_height_mm=10, 
                            base_thickness_mm=5, output_file="aruco_3d.stl"):
    """
    Gera um arquivo STL 3D de um marcador ArUco em alto relevo
    
    Args:
        marker_id: ID do marcador (0-249 para DICT_6X6_250)
        size_cm: Tamanho total em cm
        relief_height_mm: Altura do relevo em mm
        base_thickness_mm: Espessura da base em mm
        output_file: Nome do arquivo de saída
    """
    
    # Configurações
    CELLS = 6  # 6x6 cells
    BORDER_CELLS = 1  # Borda de 1 célula
    TOTAL_CELLS = CELLS + 2 * BORDER_CELLS
    
    # Criar marcador ArUco com OpenCV
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 
                                                TOTAL_CELLS * 10)  # Resolução
    
    # Converter para array numpy
    marker_array = (marker_image == 0).astype(int)  # 0=preto, 255=branco
    
    # Dimensões em mm
    total_size_mm = size_cm * 10
    cell_size_mm = total_size_mm / TOTAL_CELLS
    
    print(f"Gerando marcador 3D ID {marker_id}")
    print(f"Tamanho total: {total_size_mm}mm ({size_cm}cm)")
    print(f"Tamanho da célula: {cell_size_mm:.1f}mm")
    print(f"Altura do relevo: {relief_height_mm}mm")
    
    # Listas para vértices e faces
    vertices = []
    faces = []
    
    # Função auxiliar para adicionar um cubo
    def add_cube(x, y, z, width, height, depth):
        """Adiciona um cubo à malha"""
        nonlocal vertices, faces
        
        # 8 vértices do cubo
        cube_verts = np.array([
            [x, y, z],
            [x + width, y, z],
            [x + width, y + height, z],
            [x, y + height, z],
            [x, y, z + depth],
            [x + width, y, z + depth],
            [x + width, y + height, z + depth],
            [x, y + height, z + depth]
        ])
        
        # 12 faces triangulares do cubo
        cube_faces = np.array([
            [0, 3, 1], [1, 3, 2],  # bottom
            [4, 5, 7], [5, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [1, 2, 6], [1, 6, 5],  # right
            [2, 3, 7], [2, 7, 6],  # back
            [3, 0, 4], [3, 4, 7]   # left
        ])
        
        # Adicionar offset baseado nos vértices existentes
        vert_offset = len(vertices)
        vertices.extend(cube_verts)
        faces.extend(cube_faces + vert_offset)
    
    # 1. Criar base (placa plana)
    add_cube(0, 0, 0, total_size_mm, total_size_mm, base_thickness_mm)
    
    # 2. Adicionar células em alto relevo
    for i in range(TOTAL_CELLS):
        for j in range(TOTAL_CELLS):
            if marker_array[i, j] == 1:  # Célula preta (alta)
                x = j * cell_size_mm
                y = i * cell_size_mm
                add_cube(x, y, base_thickness_mm, 
                        cell_size_mm, cell_size_mm, relief_height_mm)
    
    # Converter para mesh STL
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Criar e salvar mesh
    marker_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            marker_mesh.vectors[i][j] = vertices[face[j]]
    
    marker_mesh.save(output_file)
    print(f"Arquivo salvo: {output_file}")
    
    # Gerar imagem de referência
    preview_size = 400
    marker_preview = cv2.resize(marker_image, (preview_size, preview_size), 
                               interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"Aruco3D/aruco_preview_id{marker_id}.png", marker_preview)
    print(f"Preview salvo: Aruco3D/aruco_preview_id{marker_id}.png")
    
    return marker_mesh

# Exemplos de uso:
if __name__ == "__main__":
    # Gerar 3 marcadores diferentes
    generate_3d_aruco_marker(marker_id=0, size_cm=15, 
                            relief_height_mm=10, output_file="Aruco3D/aruco_id0_15cm.stl")
    
    generate_3d_aruco_marker(marker_id=42, size_cm=10, 
                            relief_height_mm=8, output_file="Aruco3D/aruco_id42_10cm.stl")
    
    generate_3d_aruco_marker(marker_id=100, size_cm=20, 
                            relief_height_mm=12, output_file="Aruco3D/aruco_id100_20cm.stl")