from djitellopy import Tello

tello = Tello()

tello.connect()
tello.takeoff()


# tello.move_left(20) # Unidade: cm (20-500)
# tello.move_right(20) # Unidade: cm (20-500)
tello.rotate_clockwise(90)
tello.rotate_counter_clockwise(90)

tello.land()
