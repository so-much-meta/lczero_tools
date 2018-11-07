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
import signal
import queue



FINISHED = False

def signal_handler(signal, frame):
    """Signal handler started after threads start"""
    global FINISHED
    print('\nCtrl+C Pressed. Exiting')
    FINISHED = True

lcztools_tmp_path = pathlib.Path("/tmp/lcztools")

def clean_uds():
    """Remove any network_* - unix domain sockets""" 
    lcztools_tmp_path.mkdir(parents=True, exist_ok=True)
    for it in lcztools_tmp_path.glob('network_*'):
        it.unlink()
        
        
class BatchProcessor(threading.Thread):
    """This is responsible for processing batches and sending responses to the clients"""
    _cuda_lock = threading.Lock()
    
    def __init__(self, zmq_context, model, network_id, queue_in, queue_out):
        super().__init__()
        self.model = model
        self.network_id = network_id
        socket_path = 'inproc://{}'.format(network_id)  ## only for telling the receiver that a batch is ready
        self.context = zmq_context
        self.socket = self.context.socket(zmq.PUSH)        
        self.socket.bind(socket_path)
        self.queue_in = queue_in
        self.queue_out = queue_out        
    
    def run(self):
        while not FINISHED:
            item = self.queue_in.get()
            if item=='POISON':
                break
            features_stack, batch_ident = item 
            with self._cuda_lock:
                pol, val = self.model(features_stack)
                pol = pol.cpu().numpy()
                val = val.cpu().numpy()
            self.queue_out.put((batch_ident, pol, val))
            self.socket.send(b'')
        print("Network {} - Exiting batch processor".format(self.network_id))
        self.socket.close()
                
class DummyBatchProcessor(BatchProcessor):
    """This doesn't actually run the model. Just for testing speed"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dummy_pol = np.random.rand(1858).astype(np.float32)
        self.dummy_val = np.random.rand(1).astype(np.float32)
    
    def run(self):
        while not FINISHED:
            item = self.queue_in.get()
            if item=='POISON':
                break
            features_stack, batch_ident = item 
            with self._cuda_lock:
                pol, val = self.model(features_stack)
            length = len(batch_ident)
            pol = np.stack([self.dummy_pol]*length)
            val = np.stack([self.dummy_val]*length)
            self.queue_out.put((batch_ident, pol, val))
            self.socket.send(b'')
        print("Network {} - Exiting batch processor".format(self.network_id))
        self.socket.close()



class Receiver(threading.Thread):
    """This is responsible for receiving messages and throwing them into a queue for the
    BatchProcessor"""
    def __init__(self, zmq_context, network_id, batch_queue, response_queue, max_batch_size, dtype):
        super().__init__()
        self.network_id = network_id
        self.batch_queue = batch_queue
        self.response_queue = response_queue
        self.dtype = dtype
        self.context = zmq_context
        self.socket = self.context.socket(zmq.ROUTER)
        socket_path = lcztools_tmp_path.joinpath('network_{}'.format(network_id))        
        self.socket.bind('ipc://{}'.format(socket_path))
        batch_socket_path = 'inproc://{}'.format(network_id)
        self.batch_socket = self.context.socket(zmq.PULL)
        self.batch_socket.connect(batch_socket_path)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.poller.register(self.batch_socket, zmq.POLLIN)
        self.batch_ident = []
        self.batch_features = []
        # self.batch_size = 1
        self.batches_processed = 0
        self.batches_processed_start = time.time()
        self.max_batch_size = 32 if not max_batch_size else max_batch_size
        assert(1 <= self.max_batch_size <= 2048)
        self.messages_received = 0
        self.messages_sent = 0
        self.elapsed_items_processed = 0  # Count of responses in current duration
    
    def queue_batch(self):
        """Put a batch into the queue for the BatchProcessor""" 
        if not self.batch_ident:
            return
        # cur_features_stack = np.stack(self.batch_features).astype(self.dtype)
        # The following optimization is about twice as fast as np.stack
        item_shape = self.batch_features[0].shape
        cur_features_stack = np.frombuffer(b''.join(f for f in self.batch_features), dtype=np.uint8).reshape(-1,*item_shape)
        cur_features_stack = cur_features_stack.astype(self.dtype)
        cur_batch_ident = self.batch_ident[:]
        while True:
            try:
                self.batch_queue.put((cur_features_stack, cur_batch_ident), timeout=1)
            except queue.Full:
                if FINISHED:
                    self.batch_ident.clear()
                    self.batch_features.clear()
                    return
                else:
                    continue
            break
        self.batch_ident.clear()
        self.batch_features.clear()
        self.batches_processed += 1

    def run(self):
        blocked = False
        poll = self.poller.poll
        recv_multipart = self.socket.recv_multipart
        batch_ident_append = self.batch_ident.append
        deserialize_features = LeelaBoard.deserialize_features
        batch_features_append = self.batch_features.append
        batch_ident = self.batch_ident
        while not FINISHED:
            elapsed = time.time() - self.batches_processed_start
            if elapsed >= 5:  # Print information about every 5 seconds
                # print('Network {} -- batch_size {}: {} bps, {} sps'.format(self.network_id, self.batch_size, 400/elapsed, 400*self.batch_size/elapsed))
                if not blocked or self.elapsed_items_processed>0:
                    print('Network {} -- {} evals/s'.format(self.network_id, self.elapsed_items_processed/elapsed))
                sys.stdout.flush()
                self.batches_processed_start = time.time()
                self.elapsed_items_processed = 0
            socks = dict(poll(1000))
            if not socks:
                # We've been blocked for 2 seconds. Let's set the batch size
                if not blocked or len(self.batch_ident):
                    print("Network {} -- BLOCKED with {} items in batch".format(self.network_id, len(self.batch_ident)))
                if not len(self.batch_ident):
                    blocked = True
                continue
            # Read any requests
            if self.socket in socks:
                try:
                    while True:
                        #check for a message, this will not block
                        *ident, msg = recv_multipart(flags=zmq.NOBLOCK)
                        self.messages_received += 1
                        if len(msg)==1:
                            if msg[0] == 1:  # hi message
                                # client_ident_set.add(ident)
                                self.batch_size = self.max_batch_size
                            elif msg[0] == 255:  # bye message
                                # client_ident_set.add(ident)
                                pass
                            self.socket.send_multipart([*ident, bytes([255])])
                            self.messages_sent += 1
                            continue            
                        batch_ident_append(ident)
                        batch_features_append(deserialize_features(msg))
                        if len(self.batch_ident)>=self.max_batch_size:
                            self.queue_batch()
                except zmq.Again as e:
                    pass
            if self.batch_queue.empty() and len(self.batch_ident):
                self.queue_batch()  # Queue the rest of the items if the current batch queue is empty
            # Respond as needed     
            if self.batch_socket in socks:
                message_in = self.batch_socket.recv() # throw away...
                response_batch_ident, pol, val = self.response_queue.get()
                for ident, policy, value in zip(response_batch_ident, pol, val):
                    result = value.tobytes() + policy.tobytes()
                    self.socket.send_multipart([*ident, result])
                    self.messages_sent += 1
                    self.elapsed_items_processed += 1
        try:
            while True:  # Empty batch queue just in case, so that batch processor doesn't hang
                self.batch_queue.get(False)
        except queue.Empty:
            pass                    
        self.batch_queue.put('POISON')  # get the batch queue thread to stop
        try:
            while True:  # Empty response queue just in case, so that batch processor doesn't hang
                self.response_queue.get(False)
        except queue.Empty:
            pass
        print("Network {} - Exiting receiver".format(self.network_id))
        self.socket.close()
        self.batch_socket.close()
           
    

class NetworkServer:
    """This class is responsible for binding and proxying the frontend,
     managing the network model, and managing the BatchProcessor and Receiver"""
    def __init__(self, context, weights_file, network_id, max_batch_size=None, half=False, dummy=False):
        super().__init__ ()
        self.context = context
        self.network_id = network_id
        self.weights_file = weights_file
        self.model = None  # won't load until run
        self.half = half
        if self.half:
            dtype = np.float16
        else:
            dtype = np.float32
        self.dummy = dummy  # Use a stubbed out batch processor that doesn't actually compute
        self.batch_queue = queue.Queue(4)
        self.response_queue = queue.Queue(256) # Responses are in batches, so it's not important if this is big
        self.receiver = Receiver(self.context, network_id, self.batch_queue, self.response_queue, max_batch_size, dtype)
        self.batch_processor = None  # won't load until run
    
    def load(self):
        if dummy:
            self.model = None
            self.batch_processor = DummyBatchProcessor(self.context, self.model, self.network_id, self.batch_queue, self.response_queue)
        else:
            with BatchProcessor._cuda_lock:
                net = load_network(backend='pytorch_cuda', filename=self.weights_file, half=self.half)
            self.model = net.model
            self.batch_processor = BatchProcessor(self.context, self.model, self.network_id, self.batch_queue, self.response_queue)

    def start(self):
        self.batch_processor.daemon = True  # To prevent it from hanging after ctrl-c
        self.batch_processor.start()
        self.receiver.start()
    
    def join(self):
        self.receiver.join()
        time.sleep(1.5) # Give the batch processor a second to cleanup

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: python -m lcztools.backend.net_server ARGS")
        print("  ARGS: [--half] <weights_file_0> [max_batch_size_0] [<weights_file_1> [max_batch_size_1]...]")
        exit(1)
    clean_uds()
    context = zmq.Context()
    tasks = []
    cur_weights_file = None
    args = sys.argv[1:]
    if '--half' in args:
        print("Using half precision")
        half = True
    else:
        print("Using single precision")
        half = False
    if '--dummy' in args:
        print("Using dummy processor")
        dummy = True
    else:
        dummy = False
    args = [arg for arg in args if arg not in ('--half', '--dummy')]
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
        tasks.append(NetworkServer(context, cur_weights_file, len(tasks), max_batch_size, half=half, dummy=dummy))
        cur_weights_file = None
        if max_batch_size is None:
            if not pathlib.Path(arg).is_file():
                raise("Bad filename: {}".format(arg))
            cur_weights_file = arg
            continue
    if cur_weights_file:
        print(cur_weights_file)
        print(args)
        tasks.append(NetworkServer(context, cur_weights_file, len(tasks), max_batch_size, half=half, dummy=dummy))
    print("Loading networks and starting server tasks")
    for task in tasks:
        task.load()
    signal.signal(signal.SIGINT, signal_handler)
    for task in tasks:
        task.start()
    print()
    print("STARTED: PRESS CTRL-C TO EXIT.")
    print()
    for task in tasks:
        task.join()
    context.term()
