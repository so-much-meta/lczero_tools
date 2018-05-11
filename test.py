# -*- coding: utf-8 -*-
import json
import sys
import os
import numpy as np


sys.path.append(os.path.expanduser('~/git/lczero_tools/src'))

# For now, if using the tensorflow backend, tfprocess is imported to build the network,
# so training/tf has to be in the Python path
sys.path.append(os.path.expanduser('~/git/leela-chess/training/tf'))

weights_file = os.path.expanduser('~/git/leela-chess/release/weights.txt.gz')

import lcztools

def json_default(obj):
    # Serialize numpy floats
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError

print("Test pytorch")
lcz_net = lcztools.load_network('pytorch', weights_file)
lcz_board = lcztools.LeelaBoard()
print(lcz_board)
policy, value = lcz_net.evaluate(lcz_board)
print('Policy: {}'.format(json.dumps(policy, default=json_default, indent=3)))
print('Value: {}'.format(value))

lcz_board.push_uci('e2e4')
print(lcz_board)
policy, value = lcz_net.evaluate(lcz_board)
print('Policy: {}'.format(json.dumps(policy, default=json_default, indent=3)))
print('Value: {}'.format(value))


print("Test tensorflow")
lcz_net = lcztools.load_network('tensorflow', weights_file)
lcz_board = lcztools.LeelaBoard()
print(lcz_board)
policy, value = lcz_net.evaluate(lcz_board)
print('Policy: {}'.format(json.dumps(policy, default=json_default, indent=3)))
print('Value: {}'.format(value))

lcz_board.push_uci('e2e4')
print(lcz_board)
policy, value = lcz_net.evaluate(lcz_board)
print('Policy: {}'.format(json.dumps(policy, default=json_default, indent=3)))
print('Value: {}'.format(value))

