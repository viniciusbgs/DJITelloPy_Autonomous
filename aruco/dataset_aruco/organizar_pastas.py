import os
import shutil
from sklearn.model_selection import train_test_split

# Configurações
images_dir = "dataset_tello/images"
labels_dir = "dataset_tello/labels"
output_dir = "dataset_tello"
val_ratio = 0.2  # porcentagem de validação

# Criar pastas de saída
for split in ["train", "val"]:
    for folder in ["images", "labels"]:
        os.makedirs(os.path.join(output_dir, split, folder), exist_ok=True)

# Listar arquivos
images = sorted(os.listdir(images_dir))
labels = sorted(os.listdir(labels_dir))

# Garantir correspondência entre imagens e labels
images_set = set([os.path.splitext(f)[0] for f in images])
labels_set = set([os.path.splitext(f)[0] for f in labels])

valid_names = images_set & labels_set  # nomes que possuem imagem e label

images = [f for f in images if os.path.splitext(f)[0] in valid_names]
labels = [f for f in labels if os.path.splitext(f)[0] in valid_names]

# Ordenar para manter consistência
images.sort()
labels.sort()

# Dividir em treino e validação
train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    images, labels, test_size=val_ratio, random_state=42
)

# Função para copiar arquivos
def copy_files(file_list, src_dir, dst_dir):
    for f in file_list:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

# Copiar arquivos para pastas correspondentes
copy_files(train_imgs, images_dir, os.path.join(output_dir, "train/images"))
copy_files(train_labels, labels_dir, os.path.join(output_dir, "train/labels"))
copy_files(val_imgs, images_dir, os.path.join(output_dir, "val/images"))
copy_files(val_labels, labels_dir, os.path.join(output_dir, "val/labels"))

print("Dataset organizado com sucesso!")
print(f"Treino: {len(train_imgs)} imagens")
print(f"Validação: {len(val_imgs)} imagens")
