import pygame
from datetime import datetime, timezone
import time

class JoystickController:
    def __init__(self, joystick) -> None:
        self.joystick = joystick
        self.joystick.init()
        self.forward_backward = 0
        self.left_right = 0
        self.camera_up_down = 0
        self.camera_left_right = 0
        self.dpad = 0
        # Define the joystick axis mappings (may need adjustment)
        self.AXIS_TRACK_FORWARD_BACKWARD = 1  # Typically, the Y axis on the left stick
        self.AXIS_TRACK_LEFT_RIGHT = 0       # Typically, the X axis on the right stick
        self.AXIS_HEAD_LEFT_RIGHT = 2       # Typically, the X axis on the right stick
        self.AXIS_HEAD_UP_DOWN = 3       # Typically, the X axis on the right stick
        self.DPAD_HAT = 0             #up down left right buttons

    def start_event_loop(self, send_command):
        while True:
            pygame.event.pump()
            self.forward_backward = -self.joystick.get_axis(self.AXIS_TRACK_FORWARD_BACKWARD)  # Invert due to joystick convention
            self.left_right = self.joystick.get_axis(self.AXIS_TRACK_LEFT_RIGHT)
            self.camera_up_down = -self.joystick.get_axis(self.AXIS_HEAD_UP_DOWN)  # Invert due to joystick convention
            self.camera_left_right = self.joystick.get_axis(self.AXIS_HEAD_LEFT_RIGHT)
            self.dpad = self.joystick.get_hat(self.DPAD_HAT)
            if (self.forward_backward != 0 and self.left_right != 0):
                send_command({"fb":self.forward_backward, 
                                "lr":self.left_right, 
                                "cud":self.camera_up_down, 
                                "clr": self.camera_left_right, 
                                "dp1":self.dpad[0],
                                "dp2": self.dpad[1],
                                "timestamp_start": str(datetime.now(timezone.utc).timestamp())})
            time.sleep(0.1)
    def process_data_to_command(self,send_command, path_clear, person_location):
        self.left_right = 0.0
        self.camera_up_down = 0.0
        self.camera_left_right = 0.0
        self.forward_backward = 0.0
        self.dpad = [0, 0]
        print(person_location)
        if (path_clear):
            self.forward_backward = 0.7
            if (person_location[0] == 0 and person_location[1] == 0):
                self.forward_backward = 0.0
            if (person_location[0] > 9):
                self.forward_backward = 0.0
                self.left_right = -0.7
            if (person_location[0] < -9):
                self.forward_backward = 0.0
                self.left_right = +0.7
        send_command({"fb":self.forward_backward, 
                            "lr":self.left_right, 
                            "cud":self.camera_up_down, 
                            "clr": self.camera_left_right, 
                            "dp1":self.dpad[0],
                            "dp2": self.dpad[1],
                            "timestamp_start": str(datetime.now(timezone.utc).timestamp())})