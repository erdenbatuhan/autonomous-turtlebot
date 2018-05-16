try:
    from ConfigParser import ConfigParser  # Python 2
except ImportError:
    from configparser import ConfigParser  # Python 3

import socket
import pickle
import numpy as np
from io import BytesIO  # Python 3


class HostConnector:

    def __init__(self):
        cp = ConfigParser()
        cp.read("./config.ini")

        self.target = cp.get("Network", "vm.addr")  # IP Address of VM
        self.target_port = int(cp.get("Network", "port.header")) + 1
        self.target_socket = None
        self.last_sent = None

    def connect_to_target(self):
        self.target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.target_socket.connect((self.target, self.target_port))

    def send_data(self, data):
        self.connect_to_target()
        serialized_data = pickle.dumps(data, protocol=2)
        self.target_socket.send(serialized_data)
        self.target_socket.close()


class HostServer:

    def __init__(self):
        cp = ConfigParser()
        cp.read("./config.ini")

        self.host = cp.get("Network", "host.addr")  # IP Address of host PC
        self.listen_port = int(cp.get("Network", "port.header"))
        self.listen_socket = None
        self.last_received = None

    def listen(self):
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket.bind((self.host, self.listen_port))
        self.listen_socket.listen(1)

    def receive_data(self):
        self.conn, self.addr = self.listen_socket.accept()
        ultimate_buffer = b''

        while True:
            receiving_buffer = self.conn.recv(1024)
            if not receiving_buffer:
                break

            ultimate_buffer += receiving_buffer

        final_buffer = np.load(BytesIO(ultimate_buffer), encoding='latin1')['frame']

        return final_buffer

