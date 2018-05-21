# -*- coding: utf-8 -*-
import json
import sys
import os
import numpy as np



import lcztools

def json_default(obj):
    # Serialize numpy floats
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError

print("Test pytorch")
lcz_net = lcztools.load_network(backend='pytorch')
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
lcz_net = lcztools.load_network(backend='tensorflow')
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

