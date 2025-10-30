import cv2
import os
import numpy as np
from cv2 import aruco
from djitellopy import Tello

# ==============================
# CONFIGURAÇÕES
# ==============================
output_dir = "dataset/images"
os.makedirs(output_dir, exist_ok=True)

target_id = 9  # ID do seu marcador ArUco
count = 0

# ==============================
# CONECTAR AO TELLO
# ==============================
tello = Tello()
tello.connect()
print(f"Conectado ao Tello | Bateria: {tello.get_battery()}%")

tello.streamon()
frame_read = tello.get_frame_read()

# ==============================
# 🎯 DETECTOR ARUCO
# ==============================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

print("Pressione [ESPAÇO] para salvar imagem | [q] para sair")

# ==============================
# LOOP PRINCIPAL
# ==============================
while True:
    frame = frame_read.frame
    if frame is None:
        continue

    # Detectar ArUco
    corners, ids, rejected = detector.detectMarkers(frame)

    detected = False
    if ids is not None:
        for corner, id in zip(corners, ids):
            if id[0] == target_id:
                detected = True
                aruco.drawDetectedMarkers(frame, [corner], np.array([[id[0]]]))
                cv2.putText(frame, f"Marcador {id[0]} DETECTADO",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if not detected:
        cv2.putText(frame, "Marcador NAO DETECTADO",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar nível da bateria
    try:
        battery = tello.get_battery()
        text = f"Battery: {battery}%"
    except Exception:
        text = "Battery: N/A"

    cv2.putText(frame, text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Mostrar o vídeo
    cv2.imshow("Tello ArUco Detector", frame)

    # Teclas
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if detected:
            filename = f"{output_dir}/frame_{count:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Foto salva: {filename}")
            count += 1
        else:
            print("Marcador não detectado. Foto não salva.")
    elif key == ord('q'):
        break

# ==============================
# ENCERRAR
# ==============================
tello.streamoff()
cv2.destroyAllWindows()
