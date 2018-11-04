import zmq
import sys
import threading
import time
from random import randint, random
import numpy as np
import os
import contextlib
from lcztools import load_network, LeelaBoard
import math
import pathlib

lcztools_tmp_path = pathlib.Path("/tmp/lcztools")

def clean_uds():
    """Remove any network_* - unix domain sockets""" 
    lcztools_tmp_path.mkdir(parents=True, exist_ok=True)
    for it in lcztools_tmp_path.glob('network_*'):
        it.unlink()
    

class ServerTask(threading.Thread):
    _cuda_lock = threading.Lock()
    """ServerTask"""
    def __init__(self, weights_file, network_id, max_batch_size=None, half=False):
        super().__init__ ()
        self.network_id = network_id
        socket_path = lcztools_tmp_path.joinpath('network_{}'.format(network_id))
        self.weights_file = weights_file
        self.model = None  # won't load until run
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        print("Network ID {} with weights file {} binding to {}".format(network_id, weights_file, socket_path))
        self.socket.bind('ipc://{}'.format(socket_path))
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.batch_ident = []
        self.batch_features = []
        self.batch_size = 1
        self.batches_processed = 0
        self.batches_processed_start = time.time()
        self.max_batch_size = 256 if not max_batch_size else max_batch_size
        assert(1 <= self.max_batch_size <= 2048)
        self.finished = False
        self._send_condition = threading.Condition()
        self._process_and_send_thread = threading.Thread(target=self.process_and_send)
        self._process_and_send_thread.start()
        self.half = half
        if self.half:
            self.dtype = np.float16
        else:
            self.dtype = np.float32
        
    def process_and_send(self):
        with self._send_condition:
            while True:
                self._send_condition.wait()
                with self._cuda_lock:
                    pol, val = self.model(self.cur_features_stack)
                pol = pol.cpu().numpy()
                val = val.cpu().numpy()
                # pol, val = process_batch(net.model, batch_features)
                for ident, policy, value in zip(self.cur_batch_ident, pol, val):
                    result = value.tobytes() + policy.tobytes()
                    self.socket.send_multipart([ident, result])
    
    def process_batch(self):
        if not self.batch_ident:
            return
        with self._send_condition:
            self.cur_features_stack = np.stack(self.batch_features).astype(self.dtype)
            self.cur_batch_ident = self.batch_ident[:]
            self._send_condition.notify()
        self.batch_ident.clear()
        self.batch_features.clear()
        self.batches_processed += 1
        if (self.batches_processed % 400)==0:
            elapsed = time.time() - self.batches_processed_start
            print('Network {} -- batch_size {}: {} bps, {} sps'.format(self.network_id, self.batch_size, 400/elapsed, 400*self.batch_size/elapsed))
            sys.stdout.flush()
            self.batches_processed_start = time.time()
            
    def load(self):
        with self._cuda_lock:
            net = load_network(backend='pytorch_cuda', filename=self.weights_file, half=self.half)
        self.model = net.model
    
    def run(self):
        blocked = False
        poll = self.poller.poll
        recv_multipart = self.socket.recv_multipart
        batch_ident_append = self.batch_ident.append
        deserialize_features = LeelaBoard.deserialize_features
        batch_features_append = self.batch_features.append
        batch_ident = self.batch_ident
        while not self.finished:
            socks = poll(2000)
            # Only one socket event, so we'll not worry about doing dict(poller.poll())
            if not socks:
                # We've been blocked for 2 seconds. Let's set the batch size
                if not blocked:
                    print("Network {} -- BLOCKED with {} items in batch".format(self.network_id, len(self.batch_ident)))
                if not len(self.batch_ident):
                    blocked = True    
                if len(self.batch_ident) == 0:
                    self.batch_size = 1
                else:
                    self.batch_size = 2**int(math.log(len(self.batch_ident), 2))
                self.process_batch()
                continue
            blocked = False
            ident, msg = recv_multipart()
            if len(msg)==1:
                if msg[0] == 1:  # hi message
                    # client_ident_set.add(ident)
                    self.batch_size = self.max_batch_size
                elif msg[0] == 255:  # bye message
                    # client_ident_set.add(ident)
                    pass
                continue
            batch_ident_append(ident)
            batch_features_append(deserialize_features(msg))
            if len(batch_ident)>=self.batch_size:
                self.process_batch()
        self.socket.close()
        self.context.term()            

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: python -m lcztools.backend.net_server ARGS")
        print("  ARGS: [--half] <weights_file_0> [max_batch_size_0] [<weights_file_1> [max_batch_size_1]...]")
        exit(1)
    clean_uds()
    tasks = []
    cur_weights_file = None
    args = sys.argv[1:]
    if '--half' in args:
        print("Using half precision")
        half = True
    else:
        print("Using single precision")
        half = False
    args = [arg for arg in args if arg!='--half']
    for arg in args:
        if not cur_weights_file:
            if not pathlib.Path(arg).is_file():
                raise("Bad filename: {}".format(arg))
            cur_weights_file = arg
            continue
        max_batch_size = None
        try:
            max_batch_size = int(arg)
        except:
            pass
        tasks.append(ServerTask(cur_weights_file, len(tasks), max_batch_size, half=half))
        cur_weights_file = None
        if max_batch_size is None:
            if not pathlib.Path(arg).is_file():
                raise("Bad filename: {}".format(arg))
            cur_weights_file = arg
            continue
    if cur_weights_file:
        tasks.append(ServerTask(cur_weights_file, len(tasks), None, half=half))
    print("Loading networks and starting server tasks")
    for task in tasks:
        task.load()
    for task in tasks:
        task.start()
    for task in tasks:
        task.join()
