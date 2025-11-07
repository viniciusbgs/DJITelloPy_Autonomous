import cv2
import os

img_dir = "dataset_yolo/train/images"
label_dir = "dataset_yolo/train/labels"

for img_file in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(label_dir, img_file.rsplit(".", 1)[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    h, w, _ = image.shape

    with open(label_path, "r") as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())
            x *= w
            y *= h
            bw *= w
            bh *= h
            x1 = int(x - bw / 2)
            y1 = int(y - bh / 2)
            x2 = int(x + bw / 2)
            y2 = int(y + bh / 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("YOLO Boxes", image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
