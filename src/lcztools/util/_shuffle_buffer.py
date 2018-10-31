'''Maybe use this to help with training...'''

import random as _random

class ShuffleBufferEmptyException(Exception):
    pass


class ShuffleBufferFullException(Exception):
    pass


class ShuffleBuffer:
    '''A buffer that pops random items'''
    
    def __init__(self, length):
        self.buffer = [None]*length
        self.used = 0
        
    def full(self):
        return self.used == len(self.buffer)
    
    def push(self, item):
        if self.used == len(self.buffer):
            raise ShuffleBufferFullException
        self.buffer[self.used] = item
        self.used += 1
        
    def pop(self):
        if self.used == 0:
            raise ShuffleBufferEmptyException
        index = _random.randrange(self.used)
        item = self.buffer[index]
        self.used -= 1
        if self.used == index:
            # Just remove the last element
            self.buffer[self.used] = None
        else:
            # Move the last element to index
            self.buffer[index] = self.buffer[self.used]
            self.buffer[self.used] = None
        return item
        