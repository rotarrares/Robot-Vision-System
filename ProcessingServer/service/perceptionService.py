import torch
from model.drivePath import DrivePath
from model.floorPredictions import FloorPredictions
from data.labeling import nonFloorLabels
from shapely.geometry import box
import copy

class PerceptionService:
    def __init__(self, floor_confidence_treshold, object_confidence_treshold, device, model1) -> None:
        self.floor_confidence_treshold = floor_confidence_treshold
        self.object_confidence_treshold = object_confidence_treshold
        self.objects = []
        self.processed_objects = []
        self.device = device
        self.model1 = model1
        pass
    
    def find_objects(self, message):
        message.device = self.device
        message.model1 = self.model1
        results = message.process_objects()
        self.objects = copy.deepcopy(results)
        for result in results:
            mask = result.boxes.data[:, 4] >= self.object_confidence_treshold
            result.boxes.data = result.boxes.data[mask]
        return results
    def find_floor(self, message):
        message.device = self.device
        message.model1 = self.model1
        masks = message.process_floor()
        masks = masks.to(self.device)

        # Create a tensor filled with the background label (0)
        predictions = torch.full((message.height, message.width), 0, dtype=torch.int64, device=self.device)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions: {predictions}")

        # Iterate over each mask
        for i in range(len(masks)):
            mask = masks[i]
            
            # Resize the mask to match the predictions tensor shape
            mask_resized = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0), 
                size=(message.height, message.width), 
                mode='bilinear', 
                align_corners=False
            )
            mask_resized = mask_resized.squeeze().bool()
            
            # Assign the class label to the corresponding pixels
            predictions[mask_resized] = 0  # Assuming class label 0 represents the drivable area

        # Convert to numpy array while maintaining 2D shape
        predictions_np = predictions.cpu().numpy()
        
        # Ensure we maintain 2D shape (height, width)
        if len(predictions_np.shape) == 1:
            predictions_np = predictions_np.reshape(message.height, message.width)
            
        num_labels = 2  # Background and drivable area
        return FloorPredictions(predictions_np, num_labels)
    
    def compute_driveable_path_coords(self, img):
        return DrivePath(img)
    
    def is_path_driveable(self, img, floor):
        if (type(floor) == str):
            return False
            
        # Get the predictions and ensure they maintain 2D shape
        floor_predictions = floor.get_predictions()
        
        if len(floor_predictions.shape) != 2:
            print(f"Warning: Unexpected predictions shape {floor_predictions.shape}")
            return False
            
        print(f"Floor predictions shape: {floor_predictions.shape}")
        matrix_height, matrix_width = floor_predictions.shape
        drive_path = DrivePath(img).toMatrixCoords(matrix_width, matrix_height)
        return self.__check_drive_path_matrix(drive_path, floor_predictions)

    def find_path_obstructing_objects(self, img):
        obstructs_path = False
        drive_path = DrivePath(img)
        self.processed_objects = self.__process_object_boxes(self.objects)
        for bbox in self.processed_objects:
            if bbox[0].intersects(drive_path.polygon):
                obstructs_path = True
        return obstructs_path

    def find_object_location_relative_to_camera_center(self, img, objects_list):
        image_x = 0
        image_y = 0
        x = 0
        y = 0
        if (len(self.processed_objects) == 0) or (len(objects_list) == 0):
            return (0,0)
        maxSizeObject = next((obj for obj in self.processed_objects if obj[1] in objects_list), None)
        if (not maxSizeObject): return (0,0)
        for bbox in self.processed_objects:
            if bbox[1] in objects_list:
                if (bbox[0].area > maxSizeObject[0].area):
                    maxSizeObject = bbox
        x = maxSizeObject[0].centroid.x
        y = maxSizeObject[0].centroid.y
        height, width = img.shape[:2]
        image_x = width / 2
        image_y = height / 2
        return ((int((image_x - x) / 2), int((image_y - y) / 2)))

    def __process_object_boxes(self, objects):
        boxes = []
        for object in objects:
            names = object.names
            if (object.boxes and len(object.boxes)):
                bboxes = object.boxes.xyxy[:, :4].tolist()
                classes = object.boxes.cls.to('cpu').int().tolist()
                probs = object.boxes.conf.to('cpu').tolist()
                # Convert bounding box to a shapely polygon object
                for i in range(0, len(bboxes)):
                    if (probs[i] > self.object_confidence_treshold):
                        bbox_polygon = box(*bboxes[i])  # Unpacks the bbox list into the box function
                        boxes.append((bbox_polygon, classes[i]))
        return boxes

    def __check_drive_path_matrix(self,drive_path, floor_matrix):
        
        pt1 = drive_path.end_left  # Top-left
        pt2 = drive_path.end_right  # Top-right
        pt3 = drive_path.start_right  # Bottom-right
        pt4 = drive_path.start_left  # Bottom-left
        x1 = pt1[0]
        x2 = pt2[0]
        x3 = pt3[0]
        x4 = pt4[0]
        floor_space = True
        """
        # Create a matrix filled with zeros
        matrix = np.zeros((drive_path.height, drive_path.width, 3), dtype=np.uint8)
        # Fill the entire matrix with green color first
        matrix[:, :] = [255, 255, 255]  # RGB for green

        for i in range(0, len(floor_matrix)):
            for j in range(0, len(floor_matrix[i])):
                if (floor_matrix[i][j] == 0):
                    matrix[i][j] = [0, 255, 255]  
                else:
                    matrix[i][j] = [0, 0, 0]  
        """
        for x in range(min(x1, x4), max(x2, x3) + 1):
            top_y = int(round(self.__interpolate_y(x, pt1, pt2)))
            bottom_y = int(round(self.__interpolate_y(x, pt4, pt3)))
            for y in range(top_y, bottom_y + 1):
                #matrix[y-1][x-1] = [255, 255, 0] 
                if (floor_matrix[y-1][x-1] == 0):
                    #matrix[y-1][x-1] = [255,255, 255]
                    floor_space = False
        #matrix = cv2.resize(matrix, (500,500), interpolation=cv2.INTER_AREA)
        # Display the matrix
        #cv2.imshow('matrix', matrix)
        return floor_space

    def __interpolate_y(self, x, pt1, pt2):
        # Linear interpolation between points pt1 and pt2
        if pt1[0] == pt2[0]:  # Avoid division by zero if vertical line
            return min(pt1[1], pt2[1])
        slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        return slope * (x - pt1[0]) + pt1[1]