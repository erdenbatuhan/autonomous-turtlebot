import socket
import pickle


class VMConnector:
    def __init__(self):
        self.target = "192.168.1.29"  # IP Address of host pc
        self.target_port = 50000
        self.target_socket = None

    def connect_to_target(self):
        self.target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.target_socket.connect((self.target, self.target_port))

    def send_data(self, data):
        self.connect_to_target()
        serialized_data = pickle.dumps(data)
        self.target_socket.send(serialized_data)


class VMServer:
    def __init__(self):
        self.host = "192.168.112.129"  # IP Address of VM
        self.listen_port = 50001
        self.listen_socket = None

    def listen(self):
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket.bind((self.host, self.listen_port))
        self.listen_socket.listen(1)

    def receive_data(self):
        self.conn, self.addr = self.listen_socket.accept()
        stream = []
        while 1:
            data = self.conn.recv(4096)
            if not data:
                break
            stream.append(data)

        print(pickle.loads(b"".join(stream), encoding='latin1'))
        return pickle.loads(b"".join(stream), encoding='latin1')