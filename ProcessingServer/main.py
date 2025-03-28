from view.gui import GUI
from service.connectionService import ConnectionService
from service.perceptionService import PerceptionService
from controller.connectionController import ConnectionController
from controller.perceptionController import PerceptionController
from controller.joystickController import JoystickController
from ultralytics import YOLO
import pygame
import torch

def main():
    
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f"Using device: {device}")
  print(f" Torch cuda available: {torch.cuda.is_available()}")
  torch.device(device)
  model1 = YOLO('YoloV8_TrainModel.pt').to(device)
  HOST, PORT = "0.0.0.0", 9000
  windowName = "Robot Perception"
  floor_confidence_treshold = 0.0
  object_confidence_treshold = 0.6
  pygame.init()
  pygame.joystick.init()
  joystick = pygame.joystick.Joystick(0)
  joystick_controler = JoystickController(joystick)
  connection_service = ConnectionService(HOST, PORT)
  connection_controller = ConnectionController(connection_service)
  perception_service = PerceptionService(floor_confidence_treshold, object_confidence_treshold, device, model1)
  perception_controller = PerceptionController(perception_service)
  GUI(windowName, perception_controller, connection_controller, joystick_controler)  
if __name__ == "__main__":
  main()