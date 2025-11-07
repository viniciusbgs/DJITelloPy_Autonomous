import cv2
import os
from cv2 import aruco
import numpy as np
from djitellopy import Tello
import time

# --- Configuração Tello ---
tello = Tello()
print("Conectando ao Tello...")
try:
    tello.connect()
    tello.streamon()
    frame_read = tello.get_frame_read()
    print(f"Bateria: {tello.get_battery()}%")
    time.sleep(1)  # pequeno atraso para iniciar o stream
except Exception as e:
    print(f"Erro ao conectar ou iniciar stream do Tello: {e}")
    cv2.destroyAllWindows()
    exit()

# --- Pasta para salvar imagens ---
output_dir = "dataset_tello/images"
os.makedirs(output_dir, exist_ok=True)

# Descobrir o último índice salvo
existing_files = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
if existing_files:
    # Extrai números dos nomes dos arquivos (ex: frame_0023.jpg → 23)
    existing_indices = [
        int(f.split("_")[1].split(".")[0]) for f in existing_files if "_" in f
    ]
    count = max(existing_indices) + 1
else:
    count = 0

print(f"Iniciando a partir do índice {count:04d} (não sobrescreverá imagens existentes).")

# --- ArUco setup ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
target_id = 9

# --- Controle da bateria ---
last_battery_check = 0
battery_level = tello.get_battery()

print("Pressione [ESPACO] para salvar foto (somente se o marcador for detectado) | [q] para sair")

# --- Loop principal ---
while True:
    frame = frame_read.frame
    if frame is None:
        continue

    # Corrigir cores (Tello -> BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    original_frame = frame.copy()
    vis_frame = frame.copy()

    # Atualizar bateria a cada 5s
    current_time = time.time()
    if current_time - last_battery_check > 5:
        battery_level = tello.get_battery()
        last_battery_check = current_time

    # Detectar marcador ArUco
    corners, ids, rejected = detector.detectMarkers(vis_frame)
    is_target_detected = False

    if ids is not None:
        for corner, id in zip(corners, ids):
            if id[0] == target_id:
                aruco.drawDetectedMarkers(vis_frame, [corner], np.array([[id[0]]]))
                cv2.putText(vis_frame, f"Marcador {id[0]} DETECTADO", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                is_target_detected = True
                break
    if not is_target_detected:
        cv2.putText(vis_frame, "Marcador NAO DETECTADO", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar bateria na tela
    cv2.putText(vis_frame, f"Bateria: {battery_level}%", (vis_frame.shape[1] - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Mostrar vídeo
    cv2.imshow("Deteccao ArUco Tello", vis_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if is_target_detected:
            filename = f"{output_dir}/frame_{count:04d}.jpg"
            cv2.imwrite(filename, original_frame)
            print(f" Foto salva: {filename}")
            count += 1
        else:
            print("Marcador não detectado. Foto NÃO salva.")
    elif key == ord('q'):
        break

# --- Encerramento ---
print("Encerrando...")
tello.streamoff()
tello.end()
cv2.destroyAllWindows()
print("Sessão encerrada.")
