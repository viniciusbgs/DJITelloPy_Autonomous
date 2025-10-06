import cv2
from djitellopy import Tello
from ultralytics import YOLO

# Carregar modelo YOLOv8n (leve, ideal para rodar no Mac M2 em tempo real)
model = YOLO("models/yolov8n.pt")

# Inicializar Tello
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

# Ativar câmera
tello.streamon()
frame_read = tello.get_frame_read()

while True:
    frame = frame_read.frame
    if frame is not None:
        # === YOLOv8n: detecção de objetos ===
        results = model.predict(frame, imgsz=320, device="mps")  # "mps" usa GPU do Mac M1/M2
        annotated_frame = results[0].plot()

        # Mostrar janela
        cv2.imshow("Tello + YOLOv8n", annotated_frame)

    # Tecla 'q' fecha a janela
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()
