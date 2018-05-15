'''
This script just ensures that the engine is equal to the python implementation in evaluation
If no exception is thrown, it has passed
'''

import os
import sys
sys.path.append(os.path.expanduser('~/git/lczero_tools/src/'))
import lcztools
from lcztools.testing.leela_engine import LCZEngine
import numpy as np
import chess.pgn


engine_path = os.path.expanduser('~/git/leela-chess/release/lczero')
weights_file = os.path.expanduser('~/git/leela-chess/release/weights.txt.gz')
engine = LCZEngine(engine_path, weights_file)
board = lcztools.LeelaBoard()
engine.evaluate(board.lcz_to_uci_engine_board())
net = lcztools.load_network('pytorch', weights_file)


def eval_equal(net, engine, lczboard, tolerance=.00006):
    npol, nv = net.evaluate(lczboard)
    epol, ev = engine.evaluate(lczboard.lcz_to_uci_engine_board())
    for uci in npol:
        if abs(npol[uci] - epol[uci]) > tolerance:
            return False
    if (ev is not None) and (abs(nv - ev) > tolerance):
        return False
    return True

numgames = 200
for gamenum in range(numgames):
    print("Playing game {}/{}".format(gamenum+1,numgames))
    board = lcztools.LeelaBoard()
    counter = 500
    while (counter>0) and (not board.is_game_over()):
        counter -= 1
        if board.can_claim_draw():
            counter = max(counter, 10)
        if not eval_equal(net, engine, board):
            raise Exception("Not equal:", ' '.join(m.uci() for m in board.move_stack))
        policy, value = net.evaluate(board)
        ucis = list(policy)
        pol_values = np.fromiter(policy.values(), dtype=np.float32)
        pol_values = pol_values/pol_values.sum()
        pol_index = np.random.choice(len(pol_values), p=pol_values)
        uci = ucis[pol_index]
        #print()
        #print(board)
        #print(policy)
        #print(value)
        board.push_uci(uci)
    print()
    game = chess.pgn.Game.from_board(board.lcz_to_board())
    print(game)
    print(board)
    print("All Evals Equal: PASS")
print("Network looks good!")