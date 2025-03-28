import datetime
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


class CameraFrame:
    def __init__(self, image_data, timestamp=None, width=1280, height=720):
        self.__image_data = image_data  # The raw JPEG data as a byte array
        self.__timestamp = timestamp if timestamp else datetime.datetime.now()
        self.width = width
        self.height = height
        self.device = None
        self.model1 = None

        # Check if CUDA is available, and set the device accordingly

    def get_timestamp(self):
        return self.__timestamp
    
    def process_objects(self):
        """
        Process the JPEG data.
        This could involve decoding the JPEG, analyzing the image, etc.
        """

        #nparr = np.frombuffer(self.__image_data, np.uint8)
        #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert your image to a PyTorch tensor and send it to the device
        #img_tensor = torch.from_numpy(img).to(self.device)

        # Add batch dimension if required (YOLO models typically expect batches)
        #if len(img_tensor.shape) == 3:
        #    img_tensor = img_tensor.unsqueeze(0)
            
        # Perform inference
        #results = self.model1.predict(source=img)
        #return results  # Implement your processing logic here


    
    def process_floor(self):
        """
        Process the JPEG data for floor estimation using YOLO v8 model.
        This involves decoding the JPEG, converting it into a format suitable for the YOLO model, etc.
        """

        # Decode the JPEG image data
        nparr = np.frombuffer(self.__image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.width = img.shape[1]
        self.height = img.shape[0]

        # Perform inference using YOLO v8 model
        results = self.model1.predict(source=img) 
        # Return the masks tensor
        return results[0].masks.data

    def toCvImage(self):
        nparr = np.frombuffer(self.__image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    def fisheye_to_perspective(self, img, fov=220):
        img_height, img_width = img.shape[:2]
        fov_degrees = fov  # Fisheye FOV
        fov_radians = np.deg2rad(fov_degrees)
        focal_length = (img_width / 2) / np.tan(fov_radians / 2)  # Approximate focal length

        distCoeffs = np.zeros((4, 1))  # Initial distortion coefficients
        K = np.array([[focal_length, 0, img_width / 2],
                      [0, focal_length, img_height / 2],
                      [0, 0, 1]])  # Approximate camera matrix

        # Undistort fisheye image
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distCoeffs, np.eye(3), K, (img_width, img_height), cv2.CV_32FC1)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Define perspective transformation
        dst_size = (1280, 720)
        src_pts = np.float32([[0, 0], [img_width, 0], [img_width, img_height], [0, img_height]])
        dst_pts = np.float32([[0, 0], [dst_size[0], 0], [dst_size[0], dst_size[1]], [0, dst_size[1]]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Convert undistorted image to a tensor and transfer to GPU
        undistorted_tensor = torch.from_numpy(undistorted_img).float().permute(2, 0, 1).unsqueeze(0).to('cuda')

        # Create a mesh grid for perspective transformation
        theta = torch.tensor(M[:2, :], dtype=torch.float32).unsqueeze(0).to('cuda')
        grid = F.affine_grid(theta, undistorted_tensor.size(), align_corners=False).to('cuda')
        
        # Apply perspective warp using grid_sample
        perspective_tensor = F.grid_sample(undistorted_tensor, grid, align_corners=False)

        # Convert the result back to a numpy array
        perspective_img = perspective_tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # Ensure the image is contiguous
        perspective_img = np.ascontiguousarray(perspective_img)

        success, encoded_image = cv2.imencode('.jpg', perspective_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        if not success:
            raise ValueError("Image encoding failed.")

        self.__image_data = encoded_image.tobytes()
        return perspective_img