from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time
from ultralytics import YOLO
from cv2 import aruco
import math  
from simple_pid import PID

model = YOLO("/Users/vinicius/GITHUB/DJITelloPy/autonomous_landing/aruco_yolo_model/train28/weights/best.pt")

markerSize = 15

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters =  aruco.DetectorParameters()

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

        """
        yaw: rotação (gira esquerda/direita)
        throttle: altitude (sobe/desce)
        pitch: movimento frontal (frente/trás)
        roll: movimento lateral (esquerda/direita)
        """

        # Valores de teste
        self.pid_yaw      = PID(0.20, 0, 0,setpoint=0,output_limits=(-70,70)) 
        self.pid_throttle = PID(0.25, 0, 0,setpoint=0,output_limits=(-40,40)) 
        self.pid_pitch    = PID(0.25, 0, 0,setpoint=0,output_limits=(-20,20))
        self.pid_roll     = PID(0.20, 0, 0,setpoint=0,output_limits=(-40,40))

        # pygame helpers for stable overlays
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.overlay_texts = []

        # valores originais:
        # self.pid_yaw      = PID(0.20, 0.00005, 0.01,setpoint=0,output_limits=(-100,100)) 
        # self.pid_throttle = PID(0.25, 0.00001, 0.01,setpoint=0,output_limits=(-100,50)) 
        # self.pid_pitch    = PID(0.25, 0.0002, 0.01,setpoint=0,output_limits=(-20,20))
        # self.pid_roll     = PID(0.35, 0.00005, 0.01,setpoint=0,output_limits=(-70,70))

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

    def run(self):

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

            frame = frame_read.frame

            # Deteccao de Aruco:
            # Inferência usando YOLO
            results = model.predict(source=frame, conf=0.6, iou=0.5, device="mps", imgsz=640, verbose=False)  # se quiser usar a GPU do Mac
            # annotated_frame = frame
            annotated_frame = results[0].plot()
            
            # Deteccao usando OpenCV
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejectedImgPoints = detector.detectMarkers(annotated_frame)

            # Prioridade: YOLO > Opencv
            # Se Yolo nao detectar, o OpenCV é chamado
            # --- YOLO detections ---
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[cls]

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

                        aruco.drawDetectedMarkers(annotated_frame, corners, ids)
                        cv2.drawFrameAxes(annotated_frame, mtx, dist, rvec, tvec, 10)
                        # prepare marker and camera overlay strings (store for pygame overlay)
                        # marker_pos_str = "MARKER Position x=%4.0f y=%4.0f z=%4.0f" % (tvec.tolist()[0], tvec.tolist()[1], tvec.tolist()[2])
                        
                        # Cria os textos separados para cada eixo com a cor correspondente
                        marker_x_str = f"X (vermelho): {tvec[0]:.0f}"
                        marker_y_str = f"Y (verde)  : {tvec[1]:.0f}"
                        marker_z_str = f"Z (azul)   : {tvec[2]:.0f}"

                        R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
                        R_tc = R_ct.T
                        roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)
                        marker_att_str = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
                            math.degrees(roll_marker), math.degrees(pitch_marker), math.degrees(yaw_marker)
                        )

                        pos_camera = -R_tc * np.matrix(tvec).T
                        camera_pos_str = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f" % (
                            pos_camera[0].item(), pos_camera[1].item(), pos_camera[2].item()
                        )

                        roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
                        camera_att_str = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
                            math.degrees(roll_camera), math.degrees(pitch_camera), math.degrees(yaw_camera)
                        )

                        targ_cord_x = cx
                        targ_cord_y = cy

                        # Original
                        self.xoff = int(targ_cord_x - 480)
                        self.yoff = int(540-targ_cord_y)
                        self.zoff = int(50-tvec[2]) 
                        self.roff = int(95-math.degrees(yaw_marker))
                        vTarget = np.array((self.xoff,self.yoff,self.zoff,self.roff))

                        # --- Mostrar offsets diretamente no frame ---
                        cv2.putText(annotated_frame, f"YOLO", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        cv2.putText(annotated_frame, f"xoff: {self.xoff}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(annotated_frame, f"yoff: {self.yoff}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"zoff: {self.zoff}", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(annotated_frame, f"roff: {self.roff}", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


                        if self.manual_mode == False:
                            self.yaw_velocity = int(-self.pid_yaw(self.xoff))
                            self.up_down_velocity = int(-self.pid_throttle(self.yoff))
                            self.for_back_velocity = int(self.pid_pitch(self.zoff))
                            # self.left_right_velocity = int(self.pid_roll(self.roff))

                        if -10<self.xoff<10 and -10<self.yoff<10 and -100<self.zoff<100 and self.manual_mode == False:
                        # if -15<self.xoff<15 and -15<self.yoff<15 and -90<self.zoff<90 and self.roff<10 and self.manual_mode == False:

                            uzaklik = int((0.8883*tvec[2])-3.4264)
                            print(uzaklik)
                            self.tello.move_forward(uzaklik+30)
                            inis = True
                            say = 1
                            starttime = time.time()
                            self.tello.land()
                            self.send_rc_control = False

                        # Salva na lista de overlay_texts para desenhar depois
                        self.overlay_texts = [
                            marker_x_str,
                            marker_y_str,
                            marker_z_str,
                            marker_att_str,  
                            camera_pos_str,
                            camera_att_str
                        ]        

            # # elif ids is not None:
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

            #     # 1. Defina os pontos 3D (coordenadas do marcador)
            #     half_size = markerSize / 2
            #     obj_points = np.array([
            #         [-half_size, half_size, 0],   # Top-Left (ajuste se a sua ordem for diferente)
            #         [ half_size, half_size, 0],   # Top-Right
            #         [ half_size, -half_size, 0],  # Bottom-Right
            #         [-half_size, -half_size, 0]   # Bottom-Left
            #     ], dtype=np.float32)

            #     # 2. Use solvePnP (apenas para o primeiro marcador, 'corners[0]')
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

            # #     pos_camera = -R_tc * np.matrix(tvec).T
            # #     camera_pos_str = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f" % (
            # #         pos_camera[0].item(), pos_camera[1].item(), pos_camera[2].item()
            # #     )

            # #     roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
            # #     camera_att_str = "CAMERA Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (
            # #         math.degrees(roll_camera), math.degrees(pitch_camera), math.degrees(yaw_camera)
            # #     )

            #     targ_cord_x = m
            #     targ_cord_y = n

            #     # Original
            #     self.xoff = int(targ_cord_x - 480)
            #     self.yoff = int(540-targ_cord_y)
            #     self.zoff = int(90-tvec[2]) 
            #     self.roff = int(60-math.degrees(yaw_marker))
            #     vTarget = np.array((self.xoff,self.yoff,self.zoff,self.roff))

            #     # --- Mostrar offsets diretamente no frame ---
            #     cv2.putText(annotated_frame, f"OPENCV: {self.xoff}", (130, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            #     cv2.putText(annotated_frame, f"xoff: {self.xoff}", (130, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #     cv2.putText(annotated_frame, f"yoff: {self.yoff}", (130, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #     cv2.putText(annotated_frame, f"zoff: {self.zoff}", (130, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            #     cv2.putText(annotated_frame, f"roff: {self.roff}", (130, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


            #     if self.manual_mode == False:
            #         self.yaw_velocity = int(-self.pid_yaw(self.xoff))
            #         self.up_down_velocity = int(-self.pid_throttle(self.yoff))
            #         self.for_back_velocity = int(self.pid_pitch(self.zoff))
            #         self.left_right_velocity = int(self.pid_roll(self.roff))

            #     if -15<self.xoff<15 and -15<self.yoff<15 and -90<self.zoff<90 and self.roff<10 and self.manual_mode == False:

            #         uzaklik = int((0.8883*tvec[2])-3.4264)
            #         print(uzaklik)
            #         self.tello.move_forward(uzaklik)
            #         inis = True
            #         say = 1
            #         starttime = time.time()
            #         self.tello.land()
            #         self.send_rc_control = False

            #     # Salva na lista de overlay_texts para desenhar depois
            #     self.overlay_texts = [
            #         marker_x_str,
            #         marker_y_str,
            #         marker_z_str,
            #         marker_att_str,  
            #         camera_pos_str,
            #         camera_att_str
            #     ]    

            else:
                # Nenhum marcador detectado, zera os offsets
                if self.manual_mode == False:
                    self.xoff = 0
                    self.yoff = 0
                    self.zoff = 0
                    self.roff = 0
                # clear overlay texts when no marker
                self.overlay_texts = []
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

            # Marker / camera overlay lines (top-left)
            y0 = 5
            colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,0), (0,255,0), (0,255,0)]  # cores para cada linha
            for i, line in enumerate(self.overlay_texts):
                surf = self.font.render(line, True, colors[i])
                self.screen.blit(surf, (5, y0 + i * 20))


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

            
            # Texto descritivo da direção do movimento
            direcoes = []

            if self.for_back_velocity > 0:
                direcoes.append(f"indo para FRENTE ({self.for_back_velocity})")
            elif self.for_back_velocity < 0:
                direcoes.append(f"indo para TRÁS ({self.for_back_velocity})")

            if self.left_right_velocity > 0:
                direcoes.append(f"indo para DIREITA ({self.left_right_velocity})")
            elif self.left_right_velocity < 0:
                direcoes.append(f"indo para ESQUERDA ({self.left_right_velocity})")

            if self.up_down_velocity > 0:
                direcoes.append(f"SUBINDO ({self.up_down_velocity})")
            elif self.up_down_velocity < 0:
                direcoes.append(f"DESCENDO ({self.up_down_velocity})")

            if self.yaw_velocity > 0:
                direcoes.append(f"ROTACIONANDO para a DIREITA ({self.yaw_velocity})")
            elif self.yaw_velocity < 0:
                direcoes.append(f"ROTACIONANDO para a ESQUERDA ({self.yaw_velocity})")

            if not direcoes:
                direcoes.append("PARADO")

            print("O drone está " + ", ".join(direcoes))

def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()

if __name__ == '__main__':
    main()