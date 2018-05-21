import gzip
import os

# Note: Weight loading code taken from
# https://github.com/glinscott/leela-chess/blob/master/training/tf/net_to_model.py


LEELA_WEIGHTS_VERSION = '2'

def read_weights_file(filename):
    filename = os.path.expanduser(filename)
    if '.gz' in filename:
        opener = gzip.open
    else:
        opener = open
    with opener(filename, 'r') as f:
        version = f.readline().decode('ascii')
        if version != '{}\n'.format(LEELA_WEIGHTS_VERSION):
            raise ValueError("Invalid version {}".format(version.strip()))
        weights = []
        for e, line in enumerate(f):
            line = line.decode('ascii')
            weight = list(map(float, line.split(' ')))
            weights.append(weight)
            if e == 1:
                filters = len(line.split(' '))
                print("Channels", filters)
        blocks = e - (3 + 14)
        if blocks % 8 != 0:
            raise ValueError("Inconsistent number of weights in the file")
        blocks //= 8
        print("Blocks", blocks)
    return (filters, blocks, weights)