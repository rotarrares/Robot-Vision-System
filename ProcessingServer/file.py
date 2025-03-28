import cv2
print(f"OpenCV Version: {cv2.__version__}")
print(f"CUDA Support: {cv2.cuda.getCudaEnabledDeviceCount() > 0}") 