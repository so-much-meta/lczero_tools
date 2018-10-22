'''
Update Leela Chess's network's rule50 input weights by multiplying them by a
constant coefficient

Requires:
pip install fire

Usage:
python update_rule50_weights.py filename rule50_multiplier

Creates a new filename of the form
basename__rule50__mult.ext

E.g.,
python update_rule50_weights.py weights_345.txt.gz 0
'''


import gzip
import os
from itertools import chain
import fire


LEELA_WEIGHTS_VERSION = '2'


def _update_rule50_weights(input_weights_line, rule50_multiplier):
    '''Given input weights: 192x112x3x3, modify only the following...
    
    [range(112*9*i + 109*9, 112*9*i + 110*9) for i in range(output_channels)]
    '''
    weights = input_weights_line.strip().split()
    output_channels = len(weights)//(112*9)
    assert(output_channels == len(weights)/(112*9))
    update_indices = chain(*(range(112*9*i + 109*9, 112*9*i + 110*9) for i in range(output_channels)))
    for i in update_indices:
        weights[i] = str(float(weights[i])*rule50_multiplier)
    return ' '.join(weights) + '\n'
    
    

def update_r50_weights(filename, rule50_multiplier):
    rule50_multiplier = float(rule50_multiplier)
    dirname = os.path.dirname(filename)
    basename, ext = os.path.basename(filename).split('.', 1)
    outbase = '{}__rule50__{}'.format(basename, str(rule50_multiplier).replace('.', '_'))
    out_filename = os.path.join(dirname, '{}.{}'.format(outbase, ext))
    print(out_filename)
    if '.gz' in filename:
        opener = gzip.open
    else:
        opener = open
    output_lines = []
    with opener(filename, 'r') as f:
        version = f.readline().decode('ascii')
        output_lines.append(version)
        version = version.strip()
        if version != '{}'.format(LEELA_WEIGHTS_VERSION):
            print(filename)
            raise ValueError("Invalid version {}".format(version.strip()))
        for idx, line in enumerate(f):
            line = line.decode('ascii')
            if idx==0:
                line = _update_rule50_weights(line, rule50_multiplier)
            output_lines.append(line)
    with opener(out_filename, 'w') as f:
        for line in output_lines:
            f.write(line.encode())
    print("DONE!")

            
if __name__ == '__main__':
    fire.Fire(update_r50_weights)
