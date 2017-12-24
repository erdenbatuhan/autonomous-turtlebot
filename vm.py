try:
    from ConfigParser import ConfigParser  # Python 2
except ImportError:
    from configparser import ConfigParser  # Python 3

import socket
import pickle


class VMConnector:

    def __init__(self):
        cp = ConfigParser()
        cp.read("./config.ini")

        self.target = cp.get("Network", "host.addr")  # IP Address of host pc
        self.target_port = int(cp.get("Network", "port.header"))
        self.target_socket = None

    def connect_to_target(self):
        self.target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.target_socket.connect((self.target, self.target_port))

    def send_data(self, data):
        self.connect_to_target()
        serialized_data = pickle.dumps(data)
        self.target_socket.send(serialized_data)
        self.target_socket.close()


class VMServer:

    def __init__(self):
        cp = ConfigParser()
        cp.read("./config.ini")

        self.host = cp.get("Network", "vm.addr")  # IP Address of VM
        self.listen_port = int(cp.get("Network", "port.header")) + 1
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

        return pickle.loads(b"".join(stream))
