from ultralytics import YOLO

# Modelo pré-treinado leve
model = YOLO("yolov8n.pt")

# Treino
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=416,
    batch=16,      # menor batch para Mac M2
    device="mps",
    single_cls=True
)