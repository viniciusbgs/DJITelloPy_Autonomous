from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
from ultralytics import YOLO
from cv2 import aruco
import math  
from simple_pid import PID
import matplotlib.pyplot as plt
import csv
import os
import datetime
from matplotlib import cm
import plotly.graph_objects as go

model = YOLO("/Users/vinicius/GITHUB/DJITelloPy/autonomous_landing/aruco_yolo_model/train28/weights/best.pt")

markerSize = 15


aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = aruco.DetectorParameters()

# --- Aumento de Confiança / Estabilidade ---
# 1. Refinamento de Canto (Melhora a precisão da pose)
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX 
parameters.cornerRefinementWinSize = 7 

# 2. Filtros Anti-Ruído (Reduz falsos positivos)
# Aumenta o tamanho mínimo do marcador (em relação à imagem)
parameters.minMarkerPerimeterRate = 0.05 
# Diminui o contorno máximo
parameters.maxMarkerPerimeterRate = 3.0 
# Diminui o limite de limiar adaptativo
parameters.adaptiveThreshConstant = 5.0 
# ------------------------------------------

mtx = np.load('/Users/vinicius/GITHUB/DJITelloPy/autonomous_landing/mtx.npy')
dist = np.load('/Users/vinicius/GITHUB/DJITelloPy/autonomous_landing/dist.npy')


# Speed of the drone
S = 30

# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
FPS = 30

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

#--- 180 deg rotation matrix around the x axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations (yaw)
            - W and S: Up and down.
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 30

        # Drone offset
        self.xoff = 0
        self.yoff = 0
        self.zoff = 0
        self.roff = 0

        self.manual_mode = True  # começa em manual
        self.send_rc_control = False

        # Variável para armazenar o tempo de pouso
        self.landing_start_time = None
        self.landing_time_seconds = 0.0

        # Trajetória: lista de tuplas (timestamp, x, y, z)
        self.trajectory = []
        # caminho para salvar csv/figura
        self.output_folder = os.path.expanduser('/Users/vinicius/GITHUB/DJITelloPy/autonomous_landing/drone_trajectory_outputs')
        os.makedirs(self.output_folder, exist_ok=True)

        # Ajuste final a ser usado antes do pouso (preenchido na detecção do marcador)
        self.ajuste_final = 0

        # Erros medidos após o pouso (em cm)
        self.erro_x = 0.0
        self.erro_y = 0.0

        """
        yaw: rotação (gira esquerda/direita)
        throttle: altitude (sobe/desce)
        pitch: movimento frontal (frente/trás)
        roll: movimento lateral (esquerda/direita)
        """

        # Valores de teste
        self.pid_yaw      = PID(0.15, 0, 0,setpoint=0,output_limits=(-70,70)) 
        self.pid_throttle = PID(0.25, 0, 0,setpoint=0,output_limits=(-40,40)) 
        self.pid_pitch    = PID(0.25, 0, 0,setpoint=0,output_limits=(-20,20))
        self.pid_roll     = PID(0.20, 0, 0,setpoint=0,output_limits=(-40,40))

        # pygame helpers for stable overlays
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        # self.overlay_texts = []

        # valores originais:
        # self.pid_yaw      = PID(0.20, 0.00005, 0.01,setpoint=0,output_limits=(-100,100)) 
        # self.pid_throttle = PID(0.25, 0.00001, 0.01,setpoint=0,output_limits=(-100,50)) 
        # self.pid_pitch    = PID(0.25, 0.0002, 0.01,setpoint=0,output_limits=(-20,20))
        # self.pid_roll     = PID(0.35, 0.00005, 0.01,setpoint=0,output_limits=(-70,70))

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def run(self):


        # inicializa cálculo de FPS
        prev_time = 0
        fps = 0

        self.tello.connect()
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()

        should_stop = False
        while not should_stop:

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            # Copia o frame cru
            frame = frame_read.frame.copy()

            # calcula FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time

            # escreve o valor do FPS no frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2)

            # Faz a inferência só com a imagem "limpa"
            results = model.predict(source=frame, conf=0.9, iou=0.2, device="mps", imgsz=640, verbose=False)

            # Usa um frame separado para desenhar textos e overlays
            annotated_frame = frame.copy()

            # annotated_frame = results[0].plot()
            
            # Deteccao usando OpenCV
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejectedImgPoints = detector.detectMarkers(annotated_frame)

            # Prioridade: YOLO > Opencv
            # Se Yolo nao detectar, o OpenCV é chamado
            # --- YOLO detections ---
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # label = f"{model.names[cls]} {conf:.2f}"

                    # Cor da bounding box YOLO (em BGR)
                    color = (0, 0, 0)  
                    
                    # Desenhar a caixa
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Texto da classe
                    cv2.putText(annotated_frame, "Target", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Coordenadas do centro da caixa
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # Desenhar o centro (círculo verde)
                    cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)
                    
                    # 1. Defina os pontos 3D (coordenadas do marcador)
                    half_size = markerSize / 2
                    obj_points = np.array([
                        [-half_size, half_size, 0],   # Top-Left (ajuste se a sua ordem for diferente)
                        [ half_size, half_size, 0],   # Top-Right
                        [ half_size, -half_size, 0],  # Bottom-Right
                        [-half_size, -half_size, 0]   # Bottom-Left
                    ], dtype=np.float32)

                    # 2. Pontos 2D (cantos da caixa do YOLO na imagem)
                    image_points = np.array([
                        [x1, y1],  # topo esquerdo
                        [x2, y1],  # topo direito
                        [x2, y2],  # baixo direito
                        [x1, y2]   # baixo esquerdo
                    ], dtype=np.float32)

                    # 3. Estimar pose com solvePnP
                    success, rvec, tvec = cv2.solvePnP(obj_points, image_points, mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)

                    if success:

                        # 4. Formate rvec e tvec
                        rvec = rvec.flatten()
                        tvec = tvec.flatten()

                        # aruco.drawDetectedMarkers(annotated_frame, corners, ids)
                        cv2.drawFrameAxes(annotated_frame, mtx, dist, rvec, tvec, 10)
                        # prepare marker and camera overlay strings (store for pygame overlay)
                        # marker_pos_str = "MARKER Position x=%4.0f y=%4.0f z=%4.0f" % (tvec.tolist()[0], tvec.tolist()[1], tvec.tolist()[2])
                        
                        # # Cria os textos separados para cada eixo com a cor correspondente
                        # marker_x_str = f"X (vermelho): {tvec[0]:.0f}"
                        # marker_y_str = f"Y (verde)  : {tvec[1]:.0f}"
                        # # marker_z_str = f"Z (azul)   : {tvec[2]:.0f}"

                        R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
                        R_tc = R_ct.T
                        # roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)
                        # marker_att_str = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
                        #     math.degrees(roll_marker), math.degrees(pitch_marker), math.degrees(yaw_marker)
                        # )


                        pos_camera = -R_tc * np.matrix(tvec).T
                        # camera_pos_str = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f" % (
                        #     pos_camera[0].item(), pos_camera[1].item(), pos_camera[2].item()
                        # )
                        pos = np.array(pos_camera).flatten()  # [x, y, z]

                        # Salvar posição + timestamp (sua parte)
                        ts = datetime.datetime.now().isoformat()
                        self.trajectory.append((ts, float(pos[0]), float(pos[1]), float(pos[2])))

                        # roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
                        # camera_att_str = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
                        #     math.degrees(roll_camera), math.degrees(pitch_camera), math.degrees(yaw_camera)
                        # )

                        targ_cord_x = cx
                        targ_cord_y = cy

                        self.xoff = int(targ_cord_x - 480)
                        self.yoff = int(540-targ_cord_y)
                        self.zoff = int(30-tvec[2]) 
                        # self.roff = int(95-math.degrees(yaw_marker))
                        vTarget = np.array((self.xoff,self.yoff,self.zoff,self.roff))

                        # --- Mostrar offsets diretamente no frame ---
                        cv2.putText(annotated_frame, f"YOLO", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        cv2.putText(annotated_frame, f"xoff: {self.xoff}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(annotated_frame, f"yoff: {self.yoff}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"zoff: {self.zoff}", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # cv2.putText(annotated_frame, f"roff: {self.roff}", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                        if self.manual_mode == False:
                            self.yaw_velocity = int(-self.pid_yaw(self.xoff))
                            self.up_down_velocity = int(-self.pid_throttle(self.yoff))
                            self.for_back_velocity = int(self.pid_pitch(self.zoff))
                            # self.left_right_velocity = int(self.pid_roll(self.roff))

                        if -15<self.xoff<15 and -25<self.yoff<25 and -30<self.zoff<30 and self.manual_mode == False:
                        # if -15<self.xoff<15 and -15<self.yoff<15 and -90<self.zoff<90 and self.roff<10 and self.manual_mode == False:

                            # Ajuste final de deslocamento antes do pouso (em cm)
                            # Salva em self.ajuste_final para registro e uso posterior
                            self.ajuste_final = int(tvec[2])
                            print(self.ajuste_final)

                            # Adiciona um ponto final na trajetória: copia do último e ajusta o valor Z
                            if len(self.trajectory) > 0:
                                last_ts, last_x, last_y, last_z = self.trajectory[-1]
                                new_ts = datetime.datetime.now().isoformat()
                                # adiciona ajuste_final ao último z
                                new_point = (new_ts, float(last_x), float(last_y), float(last_z) + float(self.ajuste_final))
                                self.trajectory.append(new_point)

                            # Move e pousa usando o ajuste final
                            try:
                                self.tello.move_forward(self.ajuste_final)
                            except Exception as e:
                                print("Erro ao mover para frente:", e)
                            inis = True
                            say = 1
                            starttime = time.time()
                            self.tello.land()

                            # Coleta de erro após o pouso
                            print("\n=== POUSO COMPLETADO ===")
                            print("Por favor, digite os valores do erro em relação ao centro do ArUco:")
                            try:
                                self.erro_x = float(input("Erro em X (cm, sendo 0 o centro): "))
                                self.erro_y = float(input("Erro em Y (cm, sendo 0 o centro): "))
                                print(f"Erro registrado: X={self.erro_x:.2f}cm, Y={self.erro_y:.2f}cm")
                            except ValueError:
                                print("Erro: valores inválidos. Usando X=0, Y=0")
                                self.erro_x = 0.0
                                self.erro_y = 0.0

                            # Após o pouso, adicionar ponto final na trajetória
                            try:
                                if len(self.trajectory) > 0:
                                    last_ts, last_x, last_y, last_z = self.trajectory[-1]
                                    new_ts_landed = datetime.datetime.now().isoformat()
                                    landed_height = float(self.tello.get_height())
                                    # x e z permanecem iguais, y passa a ser a altura do drone (esperado 0)
                                    final_point = (new_ts_landed, float(last_x), landed_height, float(last_z))
                                    self.trajectory.append(final_point)
                            except Exception as e:
                                print("Erro ao adicionar ponto final após pouso:", e)

                            self.send_rc_control = False
                            self.manual_mode = True
                            self.landing_start_time = time.time()

                            # Calcula o tempo FINAL do pouso autônomo
                            if self.landing_start_time is not None:
                                self.landing_time_seconds = time.time() - self.landing_start_time
                                print(f"Tempo de Pouso Autônomo: {self.landing_time_seconds:.2f} segundos.")
                            
                                # Cria ou adiciona no arquivo de experimentos
                                arquivo_experimentos = os.path.join(self.output_folder, "experimentos.txt")
                                with open(arquivo_experimentos, "a") as f:
                                    horario = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    f.write(f"Experimento {len(os.listdir(self.output_folder))} - {horario} - "
                                            f"Tempo de pouso: {self.landing_time_seconds:.2f} s - ")
                                    
                                self.landing_start_time = None  # Reseta para novo voo

                                self.save_trajectory_and_plot('experimento')

                        # # Salva na lista de overlay_texts para desenhar depois
                        # self.overlay_texts = [
                        #     marker_x_str,
                        #     marker_y_str,
                        #     marker_z_str,
                        #     marker_att_str,  
                        #     camera_pos_str,
                        #     camera_att_str
                        # ]        

            # elif ids is not None:
            # if np.all(ids != None):
            #     corner = corners[0][0]
            #     m = int((corner[0][0]+corner[1][0]+corner[2][0]+corner[3][0])/4)
            #     n = int((corner[0][1]+corner[1][1]+corner[2][1]+corner[3][1])/4)
            #     orta = int((corner[0][0]+corner[3][0])/2)

            #     # Desenha o canto 1 (Superior Esquerdo - deve corresponder a [-half_size, half_size, 0])
            #     cv2.circle(annotated_frame, (int(corner[0][0]), int(corner[0][1])), 5, (0, 0, 255), -1) # Vermelho (1)
            #     # Desenha o canto 2 (Superior Direito - deve corresponder a [half_size, half_size, 0])
            #     cv2.circle(annotated_frame, (int(corner[1][0]), int(corner[1][1])), 5, (0, 255, 0), -1) # Verde (2)
            #     # Desenha o canto 3 (Inferior Direito - deve corresponder a [half_size, -half_size, 0])
            #     cv2.circle(annotated_frame, (int(corner[2][0]), int(corner[2][1])), 5, (255, 0, 0), -1) # Azul (3)
            #     # Desenha o canto 4 (Inferior Esquerdo - deve corresponder a [-half_size, -half_size, 0])
            #     cv2.circle(annotated_frame, (int(corner[3][0]), int(corner[3][1])), 5, (0, 255, 255), -1) # Amarelo (4)

            #     # Pontos 3D (coordenadas do marcador)
            #     half_size = markerSize / 2
            #     obj_points = np.array([
            #         [-half_size, half_size, 0],   # Top-Left (ajuste se a sua ordem for diferente)
            #         [ half_size, half_size, 0],   # Top-Right
            #         [ half_size, -half_size, 0],  # Bottom-Right
            #         [-half_size, -half_size, 0]   # Bottom-Left
            #     ], dtype=np.float32)

            #     # SolvePnP (apenas para o primeiro marcador, 'corners[0]')
            #     image_points = corner.astype(np.float32)
            #     ret, rvec, tvec = cv2.solvePnP(obj_points, image_points, mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                
            #     # 3. Formate rvec e tvec
            #     rvec = rvec.flatten()
            #     tvec = tvec.flatten()

            #     aruco.drawDetectedMarkers(annotated_frame, corners, ids)
            #     cv2.drawFrameAxes(annotated_frame, mtx, dist, rvec, tvec, 10)
            # #     # prepare marker and camera overlay strings (store for pygame overlay)
            # #     # marker_pos_str = "MARKER Position x=%4.0f y=%4.0f z=%4.0f" % (tvec.tolist()[0], tvec.tolist()[1], tvec.tolist()[2])

            #     # Cria os textos separados para cada eixo com a cor correspondente
            #     marker_x_str = f"X (vermelho): {tvec[0]:.0f}"
            #     marker_y_str = f"Y (verde)  : {tvec[1]:.0f}"
            #     marker_z_str = f"Z (azul)   : {tvec[2]:.0f}"

            #     R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
            #     R_tc = R_ct.T
            #     roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)

            # #     marker_att_str = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
            # #         math.degrees(roll_marker), math.degrees(pitch_marker), math.degrees(yaw_marker)
            # #     )

            #     pos_camera = -R_tc * np.matrix(tvec).T
            #     # camera_pos_str = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f" % (
            #     #     pos_camera[0].item(), pos_camera[1].item(), pos_camera[2].item()
            #     # )
            #     pos = np.array(pos_camera).flatten()  # [x, y, z]

            #     # Salvar posição + timestamp (sua parte)
            #     ts = datetime.datetime.now().isoformat()
            #     self.trajectory.append((ts, float(pos[0]), float(pos[1]), float(pos[2])))


            # #     roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
            # #     camera_att_str = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
            # #         math.degrees(roll_camera), math.degrees(pitch_camera), math.degrees(yaw_camera)
            # #     )

            #     targ_cord_x = m
            #     targ_cord_y = n

            #     # Original
            #     self.xoff = int(targ_cord_x - 480)
            #     self.yoff = int(540-targ_cord_y)
            #     self.zoff = int(50-tvec[2]) 
            #     # self.roff = int(95-math.degrees(yaw_marker))
            #     vTarget = np.array((self.xoff,self.yoff,self.zoff,self.roff))

            #     # --- Mostrar offsets diretamente no frame ---
            #     # cv2.putText(annotated_frame, f"OPENCV: {self.xoff}", (130, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            #     cv2.putText(annotated_frame, f"xoff: {self.xoff}", (130, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #     cv2.putText(annotated_frame, f"yoff: {self.yoff}", (130, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #     cv2.putText(annotated_frame, f"zoff: {self.zoff}", (130, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            #     cv2.putText(annotated_frame, f"roff: {self.roff}", (130, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
            #     if self.manual_mode == False:
            #             self.yaw_velocity = int(-self.pid_yaw(self.xoff))
            #             self.up_down_velocity = int(-self.pid_throttle(self.yoff))
            #             self.for_back_velocity = int(self.pid_pitch(self.zoff))
            #             # self.left_right_velocity = int(self.pid_roll(self.roff))

            #     if -15<self.xoff<15 and -25<self.yoff<25 and -100<self.zoff<100 and self.manual_mode == False:
            #     # if -15<self.xoff<15 and -15<self.yoff<15 and -90<self.zoff<90 and self.roff<10 and self.manual_mode == False:

            #         uzaklik = int((0.8883*tvec[2])-3.4264)
            #         print(uzaklik)
            #         self.tello.move_forward(uzaklik)
            #         inis = True
            #         say = 1
            #         starttime = time.time()
            #         self.tello.land()
            #         self.send_rc_control = False
            #         self.manual_mode = True
            #         self.landing_start_time = time.time()

            #         # Calcula o tempo FINAL do pouso autônomo
            #         if self.landing_start_time is not None:
            #             self.landing_time_seconds = time.time() - self.landing_start_time
            #             print(f"Tempo de Pouso Autônomo: {self.landing_time_seconds:.2f} segundos.")
                        
            #             # Magnitude do vetor XY para o erro de distância no chão
            #             erro_centro_cm = math.sqrt(tvec[0]**2 + tvec[1]**2)
            #             print(f"Erro de distância do centro do ArUco: {erro_centro_cm:.2f} cm")


            #             # Cria ou adiciona no arquivo de experimentos
            #             arquivo_experimentos = os.path.join(self.output_folder, "experimentos.txt")
            #             with open(arquivo_experimentos, "a") as f:
            #                 horario = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            #                 f.write(f"Experimento {len(os.listdir(self.output_folder))} - {horario} - "
            #                         f"Tempo de pouso: {self.landing_time_seconds:.2f} s - "
            #                         f"Erro: {erro_centro_cm:.2f} cm\n")
                            
            #             self.landing_start_time = None  # Reseta para novo voo

            #             self.save_trajectory_and_plot('experimento')

            #     # # Salva na lista de overlay_texts para desenhar depois
            #     # self.overlay_texts = [
            #     #     marker_x_str,
            #     #     marker_y_str,
            #     #     marker_z_str,
            #     #     marker_att_str,  
            #     #     camera_pos_str,
            #     #     camera_att_str
            #     # ]    

            else:
                # Nenhum marcador detectado, zera os offsets
                if self.manual_mode == False:
                    self.xoff = 0
                    self.yoff = 0
                    self.zoff = 0
                    self.roff = 0
                # clear overlay texts when no marker
                # self.overlay_texts = []
                annotated_frame = frame 

            cv2.line(annotated_frame, (480, 0), (480, 1080), (255, 0, 0), 2)  # linha vertical do centro
            cv2.line(annotated_frame, (0, 540), (960, 540), (255, 0, 0), 2)   # linha horizontal do centro

            frame_rgb = annotated_frame
            frame_rgb = np.rot90(frame_rgb)
            frame_rgb = np.flipud(frame_rgb)
            surface = pygame.surfarray.make_surface(frame_rgb)

            # Draw video frame with pygame
            self.screen.blit(surface, (0, 0))

            # Draw overlays with pygame (stable, no flicker)
            # Battery and Mode
            battery_text = f"Bateria: {self.tello.get_battery()}%"
            mode_text = "MANUAL" if self.manual_mode else "AUTOMÁTICO"
            mode_color = (255, 0, 0) if self.manual_mode else (0, 255, 0)
            battery_surf = self.font.render(battery_text, True, (255, 255, 255))
            mode_surf = self.font.render(f"Modo: {mode_text}", True, mode_color)
            self.screen.blit(battery_surf, (5, 680))
            self.screen.blit(mode_surf, (5, 705))

            # # Marker / camera overlay lines (top-left)
            # y0 = 5
            # colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,0), (0,255,0), (0,255,0)]  # cores para cada linha
            # for i, line in enumerate(self.overlay_texts):
            #     surf = self.font.render(line, True, colors[i])
            #     self.screen.blit(surf, (5, y0 + i * 20))

            # Draw center target circle on pygame (matching original coords)
            pygame.draw.circle(self.screen, (255, 0, 0), (480, 540), 10, 2)

            # Update display and tick clock instead of time.sleep
            pygame.display.flip()
            self.clock.tick(FPS)

        # Call it always before finishing. To deallocate resources.
        self.tello.end()
        cv2.destroyAllWindows()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S
        elif key == pygame.K_SPACE:  # alterna entre manual e automático com segurança
            self.toggle_mode()

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
            self.tello.rotate_clockwise(25)
            print("takeoff")
        elif key == pygame.K_l:  # land
            not self.tello.land()
            self.send_rc_control = False

    def toggle_mode(self):
        """Toggle between manual and AUTONOMOUS with safety checks.

        - If switching to MANUAL: immediately stop the drone (send zeros) and allow manual control.
        - If switching to AUTONOMO: only enable if `send_rc_control` is True (drone is flying).
          If the drone is not flying, keep manual mode and notify the user.
        """
        # If currently in AUTONOMOUS, switch to manual and stop the drone
        if not self.manual_mode:
            self.manual_mode = True
            print("Modo MANUAL ativado")
            # zero the motion commands immediately
            self.for_back_velocity = 0
            self.left_right_velocity = 0
            self.up_down_velocity = 0
            self.yaw_velocity = 0
            # send zero velocities right away to stop movement if we have RC control
            if self.send_rc_control:
                try:
                    self.tello.send_rc_control(0, 0, 0, 0)
                except Exception as e:
                    print("Erro ao enviar comando de parada:", e)
        else:
            # currently manual -> attempt to enable AUTONOMOUS
            if not self.send_rc_control:
                # safety: don't enter AUTONOMOUS mode if drone is not flying / not under RC control
                print("Não é possível ativar AUTONOMOUS: drone não está voando (aperte 't' para decolar)")
                return
            self.manual_mode = False
            print("Modo AUTOMÁTICO ativado")
            # reset PID controllers if available (best-effort)
            for pid in (self.pid_yaw, getattr(self, 'pid_roll', None), getattr(self, 'pid_pitch', None), getattr(self, 'pid_throttle', None)):
                if pid is None:
                    continue
                try:
                    pid.reset()
                except Exception:
                    # simple-pid may not have reset in some versions; ignore if missing
                    try:
                        # try clearing integral term if attribute exists
                        pid.I_term = 0
                    except Exception:
                        pass

    def update(self):
        
        # Update routine. Send velocities to Tello.
        if self.send_rc_control:
                
            self.tello.send_rc_control(
                self.left_right_velocity,
                self.for_back_velocity,
                self.up_down_velocity,
                self.yaw_velocity
            )

    def save_trajectory_and_plot(self, nome_voo):
        """Salva a trajetória e gera gráfico interativo com Plotly"""
        # Contar número de experimentos existentes
        existing_experiments = [f for f in os.listdir(self.output_folder) if f.startswith('experimento') and f.endswith('.csv')]
        experiment_number = len(existing_experiments) + 1
        
        csv_path = os.path.join(self.output_folder, f"experimento{experiment_number}.csv")
        html_path = os.path.join(self.output_folder, f"experimento{experiment_number}.html")

        # Salvar trajetória em CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "x", "y", "z", "erro_x", "erro_y"])
            for ts, x, y, z in self.trajectory:
                writer.writerow([ts, x, y, z, self.erro_x, self.erro_y])
        print(f"Trajetória salva em: {csv_path}")

        # Converter trajetória em arrays NumPy
        traj = np.array([[float(x[1]), float(x[2]), float(x[3])] for x in self.trajectory])
        xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]

        # Obter altura atual do drone
        altura_drone = self.tello.get_height()
        print(f"Altura do drone no pouso: {altura_drone} cm")
        
        # Calcular distância euclidiana APENAS do plano horizontal (X, Y)
        # Ignorar Z da trajetória pois queremos erro no chão
        distancias = np.sqrt(xs**2 + ys**2)

        # Distância final (último ponto = pouso) - apenas no plano XY
        distancia_final = np.sqrt(xs[-1]**2 + ys[-1]**2)
        print(f"Distância final do centro do ArUco (XY): {distancia_final:.2f} cm")

        # Extrai dados e timestamps
        timestamps = [datetime.datetime.fromisoformat(t[0]) for t in self.trajectory]
        t0 = timestamps[0]
        time_seconds = np.array([(t - t0).total_seconds() for t in timestamps])
        x = [t[1] for t in self.trajectory]
        y = [t[2] for t in self.trajectory]
        z = [t[3] for t in self.trajectory]

        # ===== CRIAR FIGURA COM SUBPLOTS PLOTLY =====
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        fig = make_subplots(
            rows=1, cols=4,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}, {'type': 'scatter3d'}, {'secondary_y': True}]],
            subplot_titles=(
                "Trajetória 3D do Pouso",
                "Distância ao Longo do Tempo",
                f"Trajetória 3D com Gradiente de Tempo - Experimento {experiment_number}",
                "Erro em X e Y após Pouso"
            ),
            horizontal_spacing=0.08
        )

        # ===== SUBPLOT 1: TRAJETÓRIA 3D SIMPLES =====
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='lines+markers',
                name='Trajetória',
                line=dict(color='blue', width=4),
                marker=dict(size=4, color='blue'),
                showlegend=True
            ),
            row=1, col=1
        )

        # Adicionar ponto inicial e final
        fig.add_trace(
            go.Scatter3d(
                x=[xs[0]],
                y=[ys[0]],
                z=[zs[0]],
                mode='markers',
                name='Início',
                marker=dict(size=10, color='green', symbol='diamond'),
                showlegend=True
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter3d(
                x=[xs[-1]],
                y=[ys[-1]],
                z=[zs[-1]],
                mode='markers',
                name='Fim (Pouso)',
                marker=dict(size=10, color='red', symbol='x'),
                showlegend=True
            ),
            row=1, col=1
        )

        # ===== SUBPLOT 2: DISTÂNCIA AO LONGO DO TEMPO =====
        fig.add_trace(
            go.Scatter(
                x=time_seconds,
                y=distancias,
                mode='lines+markers',
                name='Distância do Centro',
                line=dict(color='purple', width=3),
                marker=dict(size=5),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.2)'
            ),
            row=1, col=2
        )

        # ===== SUBPLOT 3: TRAJETÓRIA 3D COM GRADIENTE DE TEMPO =====
        # Normalizar tempo para cores
        norm_time = (time_seconds - time_seconds.min()) / (time_seconds.max() - time_seconds.min())
        
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='markers',
                marker=dict(
                    size=6,
                    color=time_seconds,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Tempo (s)",
                        x=1.02,
                        len=0.3,
                        y=0.33
                    ),
                    line=dict(color='rgba(100, 100, 100, 0.5)', width=0.5)
                ),
                name='Trajetória',
                showlegend=False,
                text=[f"Tempo: {t:.2f}s<br>X: {x[i]:.1f}cm<br>Y: {y[i]:.1f}cm<br>Z: {z[i]:.1f}cm" 
                      for i, t in enumerate(time_seconds)],
                hoverinfo='text'
            ),
            row=1, col=3
        )

        # Adicionar linha conectando pontos
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='lines',
                line=dict(color='rgba(150, 150, 150, 0.4)', width=2),
                name='Caminho',
                showlegend=False
            ),
            row=1, col=3
        )

        # ===== SUBPLOT 4: ERRO EM X E Y =====
        fig.add_trace(
            go.Scatter(
                x=time_seconds,
                y=[self.erro_x] * len(time_seconds),
                mode='lines',
                name='Erro X',
                line=dict(color='red', width=3),
                yaxis='y4'
            ),
            row=1, col=4
        )

        fig.add_trace(
            go.Scatter(
                x=time_seconds,
                y=[self.erro_y] * len(time_seconds),
                mode='lines',
                name='Erro Y',
                line=dict(color='green', width=3),
                yaxis='y4'
            ),
            row=1, col=4
        )

        # ===== ATUALIZAR LAYOUTS DOS EIXOS =====
        # Subplot 1
        fig.update_xaxes(title_text="X (cm)", row=1, col=1)
        fig.update_scenes(
            xaxis=dict(title='X (cm)'),
            yaxis=dict(title='Y (cm)'),
            zaxis=dict(title='Z (cm)'),
            row=1, col=1
        )

        # Subplot 2
        fig.update_xaxes(title_text="Tempo (s)", row=1, col=2)
        fig.update_yaxes(title_text="Distância do Centro (cm)", row=1, col=2)

        # Subplot 3
        fig.update_scenes(
            xaxis=dict(title='X (cm)'),
            yaxis=dict(title='Y (cm)'),
            zaxis=dict(title='Z (cm)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            row=1, col=3
        )

        # Subplot 4
        fig.update_xaxes(title_text="Tempo (s)", row=1, col=4)
        fig.update_yaxes(title_text="Erro (cm)", row=1, col=4)

        # ===== ATUALIZAR LAYOUT GERAL =====
        fig.update_layout(
            title_text=f"Análise de Pouso Autônomo - Experimento {experiment_number}",
            height=700,
            width=2600,
            showlegend=True,
            hovermode='closest',
            font=dict(size=12)
        )

        # ===== SALVAR GRÁFICO INTERATIVO =====
        fig.write_html(html_path)
        print(f"Gráfico interativo salvo em: {html_path}")
        print(f"Abra o arquivo no navegador para interagir: file://{html_path}")

        
def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()

if __name__ == '__main__':
    main()