"""Conta e imprime o número de arquivos de imagens e labels lidos nas pastas train/val do dataset_yolo.
Adicional: verifica se há valores negativos em qualquer token numérico dos arquivos de label.
"""
import os
import glob
import re

root = os.path.join(os.path.dirname(__file__), "dataset_yolo")
train_imgs = glob.glob(os.path.join(root, 'train', 'images', '*'))
train_labels = glob.glob(os.path.join(root, 'train', 'labels', '*.txt'))
val_imgs = glob.glob(os.path.join(root, 'val', 'images', '*'))
val_labels = glob.glob(os.path.join(root, 'val', 'labels', '*.txt'))

print(f"train images: {len(train_imgs)}")
print(f"train labels: {len(train_labels)}")
print(f"val images:   {len(val_imgs)}")
print(f"val labels:   {len(val_labels)}")
print(f"total images: {len(train_imgs)+len(val_imgs)}")
print(f"total labels: {len(train_labels)+len(val_labels)}")

# Optionally list a few files
if len(train_labels) > 0:
    print('\nExemplo de label (train):', os.path.basename(train_labels[0]))
if len(val_labels) > 0:
    print('Exemplo de label (val):', os.path.basename(val_labels[0]))


def find_negative_values(label_paths):
    """Procura por valores negativos convertendo tokens para float.
    Retorna uma lista de tuplas (file, line_no, token_str, token_value)."""
    negatives = []
    # regex to find potential numeric tokens (including negatives)
    token_re = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", re.IGNORECASE)
    for p in label_paths:
        try:
            with open(p, 'r') as fh:
                for i, line in enumerate(fh, start=1):
                    # find all numeric-looking tokens
                    for m in token_re.finditer(line):
                        tok = m.group(0)
                        try:
                            val = float(tok)
                        except Exception:
                            continue
                        if val < 0:
                            negatives.append((p, i, tok, val))
        except Exception:
            # ignore unreadable files but record as message
            negatives.append((p, 0, 'file_unreadable', None))
    return negatives


all_label_paths = train_labels + val_labels
neg = find_negative_values(all_label_paths)
if neg:
    print('\nFound negative numeric values in label files:')
    for f, ln, tok, val in neg:
        if ln == 0:
            print(f"  {f}: unreadable file")
        else:
            print(f"  {os.path.relpath(f)}:{ln} -> token='{tok}' value={val}")
    print(f"Total negative tokens found: {len(neg)}")
else:
    print('\nNo negative numeric tokens found in label files.')
