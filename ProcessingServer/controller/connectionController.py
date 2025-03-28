
class ConnectionController:
    def __init__(self, service):
        self.service = service

    def listen(self, drawFrame, startCommandServer):
        try:
            self.service.listen(drawFrame, startCommandServer)
        except KeyboardInterrupt:
            print("Server is shutting down.")
        finally:
            self.service.close_socket()
    
    def send(self, command):
        self.service.send(command)