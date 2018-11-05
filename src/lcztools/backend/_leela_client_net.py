"""A client of net_server.server"""
from lcztools import LeelaBoard
import zmq
import sys
import threading
import time
from random import randint, random
import numpy as np
import os
import pathlib
from itertools import count

from lcztools.backend import LeelaNetBase

lcztools_tmp_path = pathlib.Path("/tmp/lcztools")

class LeelaClientNet(LeelaNetBase):
    registered_clients = 0
    _lock = threading.Lock()
    
    @classmethod
    def register_client_id(cls):
        """Get a client ID based on PID and an incrementing value"""
        pid = os.getpid()
        with cls._lock:
            cls.registered_clients += 1
            return '{}-{}'.format(pid, cls.registered_clients)
        
    def __init__(self, policy_softmax_temp = 1.0, network_id=None, hi=True):
        super().__init__(policy_softmax_temp=policy_softmax_temp)
        network_id = 0 if network_id is None else network_id
        self.identity = self.register_client_id()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.identity = self.identity.encode('ascii')
        socket_path = lcztools_tmp_path.joinpath('network_{}'.format(network_id))
        if not pathlib.Path(socket_path).exists():
            for cnt in count():
                if cnt%30==0:
                    print("Socket {} does not exist\nPlease start network server for network_id = {}".format(socket_path, network_id))
                time.sleep(0.1)
                if pathlib.Path(socket_path).exists():
                    break
        self.socket.connect('ipc://{}'.format(socket_path))
        if hi:
            self.hi()
        print("Connected to network server {}".format(network_id))

    def call_model_eval(self, leela_board):
        message = leela_board.serialize_features()
        self.socket.send(message)
        response = self.socket.recv()
        response = memoryview(response)
        if len(response)==7436: # single precision
            value = np.frombuffer(response[:4], dtype=np.float32)
            policy = np.frombuffer(response[4:], dtype=np.float32)
        elif len(response)==3718: # half precision
            value = np.frombuffer(response[:2], dtype=np.float16)
            policy = np.frombuffer(response[2:], dtype=np.float16)            
        return policy, value
    
    def hi(self):
        """Tell the server we're here, and it should be expecting some messages"""
        self.socket.send(bytes([1]))

    def bye(self):
        """Tell the server we're going away for a bit or forever, until we hi again"""
        self.socket.send(bytes([255]))
    
    def close(self):
        self.bye()
        self.socket.close()
        self.context.term()        
