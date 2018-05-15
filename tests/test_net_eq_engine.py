'''
This script just ensures that the engine is equal to the python implementation in evaluation
If no exception is thrown, it has passed
'''

import os
import sys
sys.path.append(os.path.expanduser('~/git/lczero_tools/src/'))
import lcztools
from lcztools.testing.leela_engine import LCZEngine


engine_path = os.path.expanduser('~/git/leela-chess/release/lczero')
weights_file = os.path.expanduser('~/git/leela-chess/release/weights.txt.gz')
engine = LCZEngine(engine_path, weights_file)
board = lcztools.LeelaBoard()
engine.evaluate(board.lcz_to_uci_engine_board())
net = lcztools.load_network('pytorch', weights_file)

net.evaluate(board)


board = lcztools.LeelaBoard()
def eval_equal(net, engine, lczboard, tolerance=.00006):
    np, nv = net.evaluate(lczboard)
    ep, ev = engine.evaluate(lczboard.lcz_to_uci_engine_board())
    for uci in np:
        if abs(np[uci] - ep[uci]) > tolerance:
            return False
    if (ev is not None) and abs(nv - ev) > tolerance:
        return False
    return True

board = lcztools.LeelaBoard()

while not board.is_game_over(claim_draw=True):
    if not eval_equal(net, engine, board):
        raise Exception('Not equal!')
    policy, value = net.evaluate(board)
    print()
    print(board)
    print(policy)
    print(value)
    board.push_uci(next(iter(policy)))
print()
print(board)
policy, value = net.evaluate(board)
print(policy)
print(value)
print("Network looks good!")
