# Robot Vision System

A computer vision system for robots to detect and recognize objects.

## Model Files

This project uses several YOLOv8 and YOLOv9 model files that are not included in this repository due to their size. You will need to download these files separately:

- yolov8l.pt (87.8 MB)
- YoloV8_TrainModel.pt (23.8 MB)
- yolov9c.pt (51.8 MB)

Place these files in the root directory of the project. Note that some model files also need to be placed in the ProcessingServer directory.

## Requirements: 
    - python 3.x
    - CUDA
    - pytorch (compatible with cuda)
    - pip install ultralytics
    - pip install transformers
    - pip install numpy
    - pip install opencv-python

## Project Structure

- **fakeClient**: Contains a test client for simulating robot vision requests
- **ProcessingServer**: Main server application for processing vision data
  - controller: Contains application controllers
  - model: Contains data models
  - service: Business logic services
  - view: View components
- **webClient**: Web interface for interacting with the vision system
  - backend: Server-side components
  - frontend: Client-side components