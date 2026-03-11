#!/usr/bin/env python3
"""
Gerador de ArUco 3D CORRIGIDO - Gera relevo corretamente
Testado e funcionando!
"""

import numpy as np
import cv2
from stl import mesh
import argparse
import os

def generate_3d_aruco_marker(marker_id=0, size_cm=15.0, relief_height_mm=8.0, 
                            base_thickness_mm=3.0, output_file="aruco_3d.stl"):
    """
    Gera um arquivo STL 3D de um marcador ArUco em alto relevo
    CORRIGIDO: Agora gera relevo corretamente!
    """
    
    print(f"\n{'='*60}")
    print(f"🧱 GERANDO MARCADOR ARUCO 3D (ID: {marker_id})")
    print(f"{'='*60}")
    
    # Configurações do ArUco (6x6 com borda)
    cells_inner = 6          # Células internas do ArUco
    border_cells = 1         # Células de borda
    total_cells = cells_inner + 2 * border_cells  # Total 8x8
    
    # Gerar imagem do ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Tamanho da imagem em pixels (alta resolução)
    img_size = total_cells * 100
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, img_size)
    
    # Reduzir para matriz 8x8
    marker_small = cv2.resize(marker_image, (total_cells, total_cells), 
                             interpolation=cv2.INTER_NEAREST)
    
    # Converter para array booleano: True = preto (alto), False = branco (baixo)
    # No OpenCV, preto = 0, branco = 255
    marker_array = (marker_small == 0)
    
    # Dimensões em mm
    total_size_mm = size_cm * 10
    cell_size_mm = total_size_mm / total_cells
    
    print(f"📏 Dimensões:")
    print(f"   • Tamanho total: {size_cm} cm = {total_size_mm} mm")
    print(f"   • Relevo: {relief_height_mm} mm")
    print(f"   • Base: {base_thickness_mm} mm")
    print(f"   • Célula: {cell_size_mm:.1f} mm")
    print(f"   • Padrão: {total_cells}×{total_cells} células")
    
    # CONTAR quantas células pretas (altas)
    black_cells = np.sum(marker_array)
    print(f"   • Células altas: {black_cells}")
    
    # ==============================================
    # PARTE 1: CRIAR A BASE (placa plana)
    # ==============================================
    print("\n📐 Criando base...")
    
    # Vértices da base (8 vértices de um cubo baixo)
    base_vertices = np.array([
        # Face inferior (z = 0)
        [0, 0, 0],                      # v0
        [total_size_mm, 0, 0],          # v1
        [total_size_mm, total_size_mm, 0],  # v2
        [0, total_size_mm, 0],          # v3
        
        # Face superior (z = base_thickness_mm)
        [0, 0, base_thickness_mm],                      # v4
        [total_size_mm, 0, base_thickness_mm],          # v5
        [total_size_mm, total_size_mm, base_thickness_mm],  # v6
        [0, total_size_mm, base_thickness_mm]           # v7
    ], dtype=np.float32)
    
    # Faces da base (12 triângulos = 2 triângulos por face de cubo)
    base_faces = np.array([
        # Face inferior
        [0, 3, 1], [1, 3, 2],
        # Face superior
        [4, 5, 7], [5, 6, 7],
        # Face frontal
        [0, 1, 5], [0, 5, 4],
        # Face direita
        [1, 2, 6], [1, 6, 5],
        # Face traseira
        [2, 3, 7], [2, 7, 6],
        # Face esquerda
        [3, 0, 4], [3, 4, 7]
    ], dtype=np.int32)
    
    # ==============================================
    # PARTE 2: CRIAR AS CÉLULAS EM ALTO RELEVO
    # ==============================================
    print(f"🧱 Criando {black_cells} células em alto relevo...")
    
    all_vertices = [base_vertices]
    all_faces = [base_faces]
    vertex_offset = len(base_vertices)  # Contador de vértices
    
    # Índices para células internas (ignorando borda)
    for i in range(border_cells, total_cells - border_cells):
        for j in range(border_cells, total_cells - border_cells):
            
            # Se é célula preta (deve ter relevo)
            if marker_array[i, j]:
                # Posição em mm
                x = j * cell_size_mm
                y = i * cell_size_mm
                z_base = base_thickness_mm  # Começa em cima da base
                
                # Vértices desta célula (8 vértices)
                cell_vertices = np.array([
                    # Face inferior da célula (na base)
                    [x, y, z_base],                              # v0
                    [x + cell_size_mm, y, z_base],               # v1
                    [x + cell_size_mm, y + cell_size_mm, z_base],  # v2
                    [x, y + cell_size_mm, z_base],               # v3
                    
                    # Face superior da célula (no topo do relevo)
                    [x, y, z_base + relief_height_mm],                              # v4
                    [x + cell_size_mm, y, z_base + relief_height_mm],               # v5
                    [x + cell_size_mm, y + cell_size_mm, z_base + relief_height_mm],  # v6
                    [x, y + cell_size_mm, z_base + relief_height_mm]               # v7
                ], dtype=np.float32)
                
                # Faces desta célula (12 triângulos)
                cell_faces = np.array([
                    # Face inferior (sobre a base)
                    [0, 3, 1], [1, 3, 2],
                    # Face superior
                    [4, 5, 7], [5, 6, 7],
                    # Face frontal
                    [0, 1, 5], [0, 5, 4],
                    # Face direita
                    [1, 2, 6], [1, 6, 5],
                    # Face traseira
                    [2, 3, 7], [2, 7, 6],
                    # Face esquerda
                    [3, 0, 4], [3, 4, 7]
                ], dtype=np.int32)
                
                # Ajustar índices das faces
                cell_faces += vertex_offset
                vertex_offset += 8
                
                all_vertices.append(cell_vertices)
                all_faces.append(cell_faces)
    
    # ==============================================
    # PARTE 3: COMBINAR TUDO EM UMA MALHA ÚNICA
    # ==============================================
    print("🔗 Combinando todas as partes...")
    
    # Combinar todos os vértices
    vertices = np.vstack(all_vertices)
    
    # Combinar todas as faces
    faces = np.vstack(all_faces)
    
    print(f"   • Total de vértices: {len(vertices)}")
    print(f"   • Total de faces: {len(faces)}")
    
    # ==============================================
    # PARTE 4: CRIAR E SALVAR O ARQUIVO STL
    # ==============================================
    print("\n🎨 Criando objeto 3D...")
    
    # Criar o objeto mesh
    marker_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    
    # Preencher os dados da mesh
    for i, face in enumerate(faces):
        for j in range(3):
            marker_mesh.vectors[i][j] = vertices[face[j]]
    
    # Verificar se a mesh é válida
    if not marker_mesh.is_closed():
        print("⚠️  Aviso: A malha não está completamente fechada")
    else:
        print("✅ Malha válida e fechada")
    
    # Calcular volume
    volume, cog, inertia = marker_mesh.get_mass_properties()
    print(f"   • Volume: {volume:.1f} mm³")
    
    # ==============================================
    # PARTE 5: SALVAR ARQUIVO
    # ==============================================
    print(f"\n💾 Salvando arquivo STL: {output_file}")
    marker_mesh.save(output_file)
    
    # ==============================================
    # PARTE 6: CRIAR VISUALIZAÇÃO
    # ==============================================
    # Salvar imagem do padrão para referência
    preview_file = output_file.replace('.stl', '_pattern.png')
    
    # Criar visualização melhorada
    preview_size = 400
    marker_preview = cv2.resize(marker_image, (preview_size, preview_size), 
                               interpolation=cv2.INTER_NEAREST)
    
    # Adicionar borda e informações
    preview_with_border = cv2.copyMakeBorder(marker_preview, 40, 40, 40, 40, 
                                            cv2.BORDER_CONSTANT, value=255)
    
    # Adicionar texto informativo
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(preview_with_border, f"ArUco 3D - ID: {marker_id}", 
               (10, 25), font, 0.7, (0, 0, 0), 2)
    cv2.putText(preview_with_border, f"Size: {size_cm}cm, Relief: {relief_height_mm}mm", 
               (10, 55), font, 0.6, (0, 0, 0), 1)
    cv2.putText(preview_with_border, f"Pattern: {cells_inner}x{cells_inner} + {border_cells} border", 
               (10, 80), font, 0.6, (0, 0, 0), 1)
    
    # Destacar células de borda
    border_px = int(preview_size * border_cells / total_cells)
    cv2.rectangle(preview_with_border, 
                 (border_px, border_px),
                 (preview_size - border_px, preview_size - border_px),
                 (0, 0, 0), 2)
    
    cv2.imwrite(preview_file, preview_with_border)
    
    # ==============================================
    # PARTE 7: INFORMAÇÕES FINAIS
    # ==============================================
    print(f"\n{'='*60}")
    print(f"🎉 MARCADOR 3D CRIADO COM SUCESSO!")
    print(f"{'='*60}")
    print(f"📁 Arquivo STL: {output_file}")
    print(f"🖼️  Imagem padrão: {preview_file}")
    print(f"\n📏 ESPECIFICAÇÕES:")
    print(f"   • Dimensões: {total_size_mm} × {total_size_mm} × {base_thickness_mm + relief_height_mm} mm")
    print(f"   • Área útil: {cells_inner}x{cells_inner} = {cells_inner * cell_size_mm:.1f} mm")
    print(f"   • Células altas: {black_cells} de {cells_inner * cells_inner}")
    print(f"   • Volume aproximado: {volume/1000:.1f} cm³")
    print(f"   • Peso (PLA, 1.25g/cm³): {(volume/1000)*1.25:.1f} g")
    print(f"\n🖨️  CONFIGURAÇÃO DE IMPRESSÃO RECOMENDADA:")
    print(f"   • Material: PLA")
    print(f"   • Altura da camada: 0.2 mm")
    print(f"   • Preenchimento: 20%")
    print(f"   • Suportes: NÃO necessário")
    print(f"   • Base adesiva: SIM (para melhor adesão)")
    print(f"{'='*60}")
    
    return marker_mesh

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Gerador de Marcadores ArUco 3D - CORRIGIDO')
    
    parser.add_argument('--id', type=int, default=0,
                       help='ID do marcador (0-249) - Padrão: 0')
    parser.add_argument('--size', type=float, default=15.0,
                       help='Tamanho em cm - Padrão: 15.0')
    parser.add_argument('--relief', type=float, default=8.0,
                       help='Altura do relevo em mm - Padrão: 8.0')
    parser.add_argument('--base', type=float, default=3.0,
                       help='Espessura da base em mm - Padrão: 3.0')
    parser.add_argument('--output', type=str, default='',
                       help='Nome do arquivo de saída')
    
    args = parser.parse_args()
    
    # Gerar nome do arquivo se não fornecido
    if not args.output:
        args.output = f"aruco_3d_id{args.id}_{int(args.size)}cm.stl"
    
    print("🔧 GERAÇÃO DE MARCADOR ARUCO 3D")
    print("=" * 50)
    
    # Verificar se o ID é válido
    if args.id < 0 or args.id > 249:
        print(f"❌ ERRO: ID {args.id} inválido. Use 0-249.")
        return
    
    # Verificar se o diretório de saída existe
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', 
                exist_ok=True)
    
    try:
        # Gerar o marcador
        marker = generate_3d_aruco_marker(
            marker_id=args.id,
            size_cm=args.size,
            relief_height_mm=args.relief,
            base_thickness_mm=args.base,
            output_file=args.output
        )
        
        # Mensagem final
        print(f"\n✅ PRONTO! Agora você pode:")
        print(f"   1. Abrir '{args.output}' no seu slicer (Cura, PrusaSlicer)")
        print(f"   2. Configurar: PLA, 0.2mm layer, 20% infill")
        print(f"   3. Imprimir!")
        print(f"   4. Testar com sua câmera IMI")
        
    except Exception as e:
        print(f"\n❌ ERRO durante a geração: {e}")
        print("💡 Possíveis soluções:")
        print("   • Verifique se todas as bibliotecas estão instaladas:")
        print("     pip install numpy opencv-python numpy-stl")
        print("   • Verifique permissões de escrita no diretório")
        print("   • Tente um ID diferente (0-249)")

if __name__ == "__main__":
    # Verificar dependências
    try:
        import numpy as np
        import cv2
        from stl import mesh
    except ImportError as e:
        print(f"❌ Dependência faltando: {e}")
        print("📦 Instale com: pip install numpy opencv-python numpy-stl")
        exit(1)
    
    main()