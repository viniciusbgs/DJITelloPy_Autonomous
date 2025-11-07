import numpy as np
import cv2
import glob
import pathlib

# Critério de refinamento dos cantos
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Configuração do tabuleiro: 8x6 vértices → 9x7 quadrados de 30mm
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) * 30  # multiplicando pelo tamanho real do quadrado em mm

# Listas para pontos 3D e 2D
objpoints = []  # pontos no espaço real
imgpoints = []  # pontos na imagem

# Diretório das imagens
images = glob.glob('/Users/vinicius/GITHUB/DJITelloPy/calibracao_camera/images/*.jpg')

# Pasta para salvar resultados (opcional)
path = '/Users/vinicius/GITHUB/DJITelloPy/calibracao_camera/resultado_calibracao'
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

found = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecta os cantos do tabuleiro
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Desenhar cantos detectados
        img = cv2.drawChessboardCorners(img, (8,6), corners2, ret)
        found += 1
        cv2.imshow('img', img)
        cv2.waitKey(500)
        cv2.imwrite(f'{path}/resultado_{found}.png', img)

print("Número de imagens usadas para calibração:", found)
cv2.destroyAllWindows()

# Calibração da câmera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("ret = ", ret)

# Salva como arquivos numpy
np.save("mtx.npy", mtx)
np.save("dist.npy", dist)

print("Arquivos mtx.npy e dist.npy salvos com sucesso!")
