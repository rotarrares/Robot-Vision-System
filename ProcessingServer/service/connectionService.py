import socket
import cv2
import socket
from model.cameraFrame import CameraFrame
from collections import defaultdict
import datetime
import json

class ConnectionService:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client_ip = ''
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_port = 9000
        try:
            self.sock.bind((self.host, self.port))
            print(f"Server started at {self.host}:{self.port}")
        except socket.error as e:
            print(f"Failed to bind socket: {e}")
            raise
        self.frames_data = defaultdict(lambda: {"chunks": {}, "total_chunks": None})
        print(f"Server started at {self.host}:{self.port}")

    def listen(self, drawFrame, startCommandServer):
        first = 0
        while True:
            data, client_address = self.sock.recvfrom(65535)  # Buffer size is 1024 bytes
            self.client_ip, self.client_port = client_address
            if first < 1:
                print(f'got message from address {client_address}')
                self.send('hy there')
                first = first + 1
                startCommandServer()
            else:
                frame_data = self.process_received_chunk(data)
                if frame_data and isinstance(frame_data, CameraFrame):  
                    drawFrame(frame_data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    

    def close_socket(self):
        self.sock.close()
    
    def send(self, data):
        if (self.client_ip and self.client_port):
            print('sending command', data)
            json_data = json.dumps(data)
            self.sock.sendto(json_data.encode(), (self.client_ip, self.client_port))

    def process_received_chunk(self, data):
        # Extract header and chunk data
        try:
            # Extract header and chunk data
            header, chunk = data.split(b';', 1)
        except ValueError as e:
            print("Malformed packet received, could not split header and chunk")
            return None

        try:
            timestamp_str, chunk_number_str, total_chunks_str = header.decode().split(',')
        except ValueError as e:
            print("Malformed header received, could not extract timestamp, chunk number, or total chunks")
            return None

        chunk_number = int(chunk_number_str)
        total_chunks = int(total_chunks_str)
        # Optionally convert the timestamp string to a datetime object, if needed
        frame_timestamp = datetime.datetime.fromisoformat(timestamp_str)
        # Store chunk and update total_chunks if necessary
        self.frames_data[frame_timestamp]["chunks"][chunk_number] = chunk
        self.frames_data[frame_timestamp]["total_chunks"] = total_chunks

        # Check if all chunks for this frame have been received
        if len(self.frames_data[frame_timestamp]["chunks"]) == total_chunks:
            # Reassemble the frame
            frame_data = b''.join(self.frames_data[frame_timestamp]["chunks"][i] for i in range(total_chunks))
            print(f"Received picture size of {len(frame_data)}")
            del self.frames_data[frame_timestamp]
            print(f'frame timestamp:{frame_timestamp}')
            return CameraFrame(frame_data, frame_timestamp)
        return None