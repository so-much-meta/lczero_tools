from lcztools import LeelaBoard
import zmq
import sys
import threading
import time
from random import randint, random
import numpy as np


class ClientTask(threading.Thread):
    """ClientTask"""
    def __init__(self, id):
        self.id = id
        threading.Thread.__init__ (self)
        self.board = LeelaBoard()
        self.board.push_uci('d2d4')

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        identity = u'client-%d' % self.id
        socket.identity = identity.encode('ascii')
        socket.connect('ipc:///tmp/lcztools/network_0')
        socket.send(bytes([1])) # Hi message
        socket.recv()
        message = self.board.serialize_features()
        for _ in range(20000):
            # print("Client {} sending message".format(identity))
            # message = self.board.serialize_features()
            for _ in range(32):
                socket.send(message)
                # print("Send:", self.id)
            
            for _ in range(32):
                response = memoryview(socket.recv())
                # print("Response:", self.id)
                # print ("Client {} received message of length {}".format(identity, len(response)))
                if len(response)==7436: # single precision
                    value = np.frombuffer(response[:4], dtype=np.float32)
                    policy = np.frombuffer(response[4:], dtype=np.float32)
                elif len(response)==3718: # half precision
                    value = np.frombuffer(response[:2], dtype=np.float16)
                    policy = np.frombuffer(response[2:], dtype=np.float16)
                # time.sleep(0.5)
        socket.close()
        context.term()

def main():
    """main function"""
    for i in range(32):
        client = ClientTask(i)
        client.start()

if __name__ == "__main__":
    main()
    