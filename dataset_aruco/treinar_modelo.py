from ultralytics import YOLO

# Modelo pré-treinado leve
model = YOLO("yolov8n.pt")

# Treino
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,      # menor batch para Mac M2
    device="mps",
    single_cls=True
)
