import cv2
import numpy as np
from cv2 import aruco

def generate_tello_aruco():
    """
    Gera o marcador ArUco compatível com seu código do Tello
    Usando DICT_ARUCO_ORIGINAL e tamanho de 15cm como no seu código
    """
    
    # Usar o mesmo dicionário do seu código
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    
    # ID do marcador (você pode mudar este valor)
    marker_id = 0  # Use IDs de 0 a 1023 para DICT_ARUCO_ORIGINAL
    
    # Tamanho em pixels para impressão de qualidade (300 DPI)
    # Para 15cm (150mm) a 300 DPI = 1772 pixels
    marker_size_pixels = 1772
    
    print(f"Gerando marcador ArUco ID: {marker_id}")
    print(f"Dicionário: DICT_ARUCO_ORIGINAL (compatível com seu código)")
    print(f"Tamanho real: 15cm x 15cm")
    print(f"Resolução: {marker_size_pixels}x{marker_size_pixels} pixels (300 DPI)")
    
    # Gerar marcador
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
    
    # Criar imagem com bordas brancas para impressão
    border_size = 200  # Borda de ~1.7cm
    final_size = marker_size_pixels + 2 * border_size
    final_image = np.ones((final_size, final_size), dtype=np.uint8) * 255
    
    # Colocar marcador no centro
    final_image[border_size:border_size+marker_size_pixels, 
                border_size:border_size+marker_size_pixels] = marker_image
    
    # Adicionar informações do marcador
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    font_thickness = 6
    
    # Texto principal
    text1 = f"ArUco Marker ID: {marker_id}"
    text2 = "Tamanho: 15cm x 15cm"
    text3 = "DICT_ARUCO_ORIGINAL"
    
    # Posicionar textos
    text_y = final_size - 150
    cv2.putText(final_image, text1, (50, text_y), font, font_scale, 0, font_thickness)
    cv2.putText(final_image, text2, (50, text_y + 80), font, font_scale, 0, font_thickness)
    cv2.putText(final_image, text3, (50, text_y + 160), font, font_scale, 0, font_thickness)
    
    # Salvar imagem
    filename = f"aruco/marker_id{marker_id}_15cm.png"
    cv2.imwrite(filename, final_image)
    
    print(f"\n✅ Marcador salvo como: {filename}")
    print(f"📐 Dimensões da imagem: {final_size}x{final_size} pixels")
    
    return final_image, filename

def generate_multiple_tello_markers():
    """
    Gera vários marcadores para testes
    """
    marker_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # IDs dos marcadores
    
    print("Gerando múltiplos marcadores para testes...")
    
    for marker_id in marker_ids:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        marker_size_pixels = 1772
        
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
        
        # Criar imagem com bordas
        border_size = 200
        final_size = marker_size_pixels + 2 * border_size
        final_image = np.ones((final_size, final_size), dtype=np.uint8) * 255
        
        final_image[border_size:border_size+marker_size_pixels, 
                    border_size:border_size+marker_size_pixels] = marker_image
        
        # Adicionar ID
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"ID: {marker_id}"
        cv2.putText(final_image, text, (50, final_size - 50), font, 3, 0, 8)
        
        filename = f"aruco/_aruco_id{marker_id}_15cm.png"
        cv2.imwrite(filename, final_image)
        print(f"✅ Marcador ID {marker_id} salvo como: {filename}")

def create_calibration_sheet():
    """
    Cria uma folha A4 com múltiplos marcadores menores para calibração
    """
    print("Criando folha de calibração A4...")
    
    # Configurações A4 (300 DPI)
    a4_width = 2480   # 210mm
    a4_height = 3508  # 297mm
    
    # Criar folha branca
    sheet = np.ones((a4_height, a4_width), dtype=np.uint8) * 255
    
    # Dicionário ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    
    # Configurações dos marcadores menores
    marker_size = 400  # Tamanho menor para caber mais na folha
    margin = 200
    spacing = 100
    
    # IDs dos marcadores
    marker_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    # Calcular posições (3x3 grid)
    cols = 3
    rows = 3
    
    for i, marker_id in enumerate(marker_ids):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        # Gerar marcador
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        
        # Calcular posição
        x = margin + col * (marker_size + spacing)
        y = margin + row * (marker_size + spacing)
        
        # Colocar na folha
        sheet[y:y+marker_size, x:x+marker_size] = marker_img
        
        # Adicionar ID
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"ID: {marker_id}"
        text_x = x + marker_size//4
        text_y = y + marker_size + 50
        cv2.putText(sheet, text, (text_x, text_y), font, 1.5, 0, 4)
    
    # Salvar folha
    filename = "aruco/tello_calibration_sheet_A4.png"
    cv2.imwrite(filename, sheet)
    print(f"✅ Folha de calibração salva como: {filename}")
    
    return sheet

if __name__ == "__main__":
    print("=== Gerador de Marcadores ArUco para Tello ===\n")
    
    # 1. Gerar marcador principal (15cm)
    print("1. Gerando marcador principal (15cm x 15cm):")
    generate_tello_aruco()
    
    print("\n" + "="*50 + "\n")
    
    # 2. Gerar múltiplos marcadores
    print("2. Gerando múltiplos marcadores para testes:")
    generate_multiple_tello_markers()
    
    print("\n" + "="*50 + "\n")
    
    # 3. Criar folha de calibração
    print("3. Criando folha de calibração A4:")
    create_calibration_sheet()
    
    print("\n" + "="*50)
    print("🎯 INSTRUÇÕES DE IMPRESSÃO:")
    print("="*50)
    print("1. Use papel branco comum (A4 ou maior)")
    print("2. Configure a impressora para 300 DPI")
    print("3. Imprima SEM redimensionamento (tamanho real)")
    print("4. Meça o marcador impresso - deve ter exatamente 15cm")
    print("5. Cole em superfície rígida para melhor detecção")
    print("6. Use o marcador ID 0 primeiro para testes")
    print("\n💡 DICA: O marcador de 15cm é o ideal para seu código!")
    print("   (markerSize = 15 no seu código)")