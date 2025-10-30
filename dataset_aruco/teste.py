import os

base_path = "dataset_yolo"

for split in ["train", "val"]:
    labels_dir = os.path.join(base_path, split, "labels")
    for file in os.listdir(labels_dir):
        if not file.endswith(".txt"):
            continue
        path = os.path.join(labels_dir, file)
        new_lines = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"[ERRO] {file}: linha inválida -> {line.strip()}")
                    continue
                try:
                    cls = int(float(parts[0]))
                    if cls < 0:
                        print(f"[ERRO] {file}: class_id negativo -> {cls}")
                        continue
                    coords = [float(x) for x in parts[1:]]
                    if not all(0 <= c <= 1 for c in coords):
                        print(f"[ERRO] {file}: coordenadas fora do intervalo [0,1] -> {coords}")
                        continue
                    # força single class
                    new_lines.append("0 " + " ".join(map(str, coords)))
                except Exception as e:
                    print(f"[ERRO] {file}: {e}")
        with open(path, "w") as f:
            f.write("\n".join(new_lines))
print("✅ Labels revisados: nenhum class_id negativo ou inválido restante!")
