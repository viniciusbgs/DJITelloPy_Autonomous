from djitellopy import Tello
import cv2
import os

tello = Tello()
tello.connect()
tello.streamon()

print("Bateria = ", tello.get_battery(), "%")

save_path = 'images'
os.makedirs(save_path, exist_ok=True)

frame_count = 0
while frame_count < 30:
    frame = tello.get_frame_read().frame
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Pressione 'c' para capturar
        filename = os.path.join(save_path, f'image2_{frame_count}.jpg')
        cv2.imwrite(filename, frame)
        print(f'Salvo {filename}')
        frame_count += 1

    elif key == 27:  # Esc para sair
        break

cv2.destroyAllWindows()
tello.streamoff()
tello.end()
