
class PerceptionController:
    def __init__(self, service):
        self.service = service
    
    def find_objects(self, frame_data):
        try:
            objects_data = self.service.find_objects(frame_data)
            return objects_data
        except:
            return f"Couldn't find objects"
    
    def find_floor(self, frame_data):
        try:
            return self.service.find_floor(frame_data)
        except:
            return "Couldn't detect floor"
        
        
    def find_obstructed_path(self, img, logits):
        path_driveable = self.service.is_path_driveable(img, logits)
        #print(f'path driveable {path_driveable}')
        path_objects = self.service.find_path_obstructing_objects(img)
        #print(f'path objects {path_objects}')
        return not (not path_driveable or path_objects)
    
    def get_driveable_path_coords(self, img):
        return self.service.compute_driveable_path_coords(img)
    
    def find_object_location(self, img, object_list):
        return self.service.find_object_location_relative_to_camera_center(img, object_list)