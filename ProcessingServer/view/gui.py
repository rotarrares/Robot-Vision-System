import cv2
import torch
import numpy as np
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class GUI:
    def __init__(self, windowName, perceptionController, connectionController, joystickController):
        self.perceptionController = perceptionController
        self.connectionController = connectionController
        self.joystickController = joystickController
        self.color_palette = None
        self.windowName = windowName
        self.connection_started = False
        self.command_sending_thread = threading.Thread(target=self.joystickController.start_event_loop, args=[self.connectionController.send])
        self.connectionController.listen(self.drawFrame, self.command_sending_thread.start)
        
        #self.joystickController.start_event_loop(self.connectionController.send)
    

    def drawFrame(self, frame_data):
        if (not self.connection_started): 
            cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
            self.connection_started = True
        image = self.processFrame(frame_data)
        cv2.imshow(self.windowName, image)

    def displayFrame(self, frame_data):
        img = frame_data.toCvImage()
        return img

    def processFrame(self, frame_data):
        img = frame_data.toCvImage()
        img = frame_data.fisheye_to_perspective(img, 270)
        #objects = self.perceptionController.find_objects(frame_data)
        floor = self.perceptionController.find_floor(frame_data)
        path_clear = self.perceptionController.find_obstructed_path(img, floor)
        #person_location = self.perceptionController.find_object_location(img, [0]) # 0 stands for person in yolov8
        self.joystickController.process_data_to_command(send_command = self.connectionController.send, path_clear = path_clear, person_location = (0, 0))
        #objected_image = self.display_object_results(img, objects)
        segmented_image = self.display_segmentation_results(img, floor)
        final_image = self.add_visual_tracks(segmented_image, path_clear)
        return final_image

    def display_segmentation_results(self, img, floor):
        if (type(floor)) == str:
            print(f'Warning :{floor}')
            return img

        # Print the shape and unique labels of the predictions
        print(f"Floor predictions shape: {floor.get_predictions().shape}")
        print(f"Unique labels in floor predictions: {np.unique(floor.get_predictions())}")

        if self.color_palette is None:
            self.color_palette = np.random.randint(0, 255, (floor.get_num_labels(), 3), dtype=np.uint8)

        color_map = self.color_palette[floor.get_predictions()]
        color_map_image = color_map.squeeze()
        height, width = img.shape[:2]
        color_map_resized = cv2.resize(color_map_image, (width, height), interpolation=cv2.INTER_NEAREST)

        # Print the shape of the color map
        print(f"Color map shape: {color_map_resized.shape}")

        original_image_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        alpha = 0.2
        overlayed_image = cv2.addWeighted(original_image_rgb, alpha, color_map_resized, 1 - alpha, 0)

        return overlayed_image
    
    
    def display_object_results(self, img, object_results):
        
        if img is not None:
            for result in object_results:
                img = result.plot(img=img, boxes=True, probs=True, labels=True, conf=True, masks=True)
        return img
    
    def add_visual_tracks(self, img, path_obstructed):
        if type(img) == str: 
            print(f'Warning :{img}')
            return img
        # Calculate starting points for the lines
        driveable_path = self.perceptionController.get_driveable_path_coords(img)
        line_color =  (20,70, 255)
        if path_obstructed:
            line_color = (255, 70, 20)
        # Draw the left line
        print(path_obstructed)
        cv2.line(img, driveable_path.start_left, driveable_path.end_left, line_color, thickness=5)  # Blue color in BGR

        # Draw the right line
        cv2.line(img, driveable_path.start_right, driveable_path.end_right, line_color, thickness=5)  # Blue color in BGR
        return img
    def run_models_concurrently(self, frame_data, img):
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Schedule the tasks and get Future objects
            future_objects = executor.submit(self.perceptionController.find_objects, frame_data)
            future_floor = executor.submit(self.perceptionController.find_floor, frame_data)

            # Wait for both futures to complete and retrieve their results
        
            future_person_location = executor.submit(self.perceptionController.find_object_location,img, [0])
            objects_result = future_objects.result()
            floor_result = future_floor.result()
            future_path_clear = executor.submit(self.perceptionController.find_obstructed_path,img, floor_result)
            path_clear = future_path_clear.result()
            person_location = future_person_location.result()
            # Now you can use objects_result and floor_result
            #print(f"Objects result: {objects_result}")
            #print(f"Floor result: {floor_result}")

            # If you need to use these results for further processing,
            # you can return them from this function or directly pass them to another function here.
            return floor_result, objects_result, path_clear, person_location