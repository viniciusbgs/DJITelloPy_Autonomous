import cv2
import os

# Caminhos
img_dir = "dataset_tello/images"
label_dir = "dataset_tello/labels"
os.makedirs(label_dir, exist_ok=True)

# Dicionário ArUco — use o mesmo que detectou seu marcador
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

target_id = 9  # <-- O aruco tem ID 9

# Loop nas imagens
for img_file in os.listdir(img_dir):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(img_dir, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Erro ao abrir {img_file}")
        continue

    h, w, _ = image.shape
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None:
        print(f"Nenhum marcador encontrado em {img_file}")
        os.remove(img_path)
        continue

    found = False
    for corner, id in zip(corners, ids):
        if id[0] != target_id:
            continue

        found = True
        x_min = corner[0][:, 0].min()
        y_min = corner[0][:, 1].min()
        x_max = corner[0][:, 0].max()
        y_max = corner[0][:, 1].max()

        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h

        # Classe 0 (único marcador)
        label_path = os.path.join(label_dir, img_file.rsplit(".", 1)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    if found:
        print(f"Marcador ID 9 encontrado em {img_file}")
    else:
        print(f"Marcadores detectados, mas nenhum com ID 9 em {img_file}")

# print("\nRótulos YOLO gerados em 'dataset/labels'")
