import cv2
from cv2 import aruco

# --- escolha o mesmo dicionário usado no seu código ---
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters()

# --- abre a câmera ---
cap = cv2.VideoCapture(0)  # 0 = webcam padrão

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecta marcadores
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)

    # Se encontrar marcadores, desenha e mostra o ID
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i, id in enumerate(ids):
            print(f"Marcador detectado com ID: {id[0]}")

    # Mostra o vídeo
    cv2.imshow("Deteccao ArUco", frame)

    # Sai com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
