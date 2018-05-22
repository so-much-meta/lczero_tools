'''
This script just ensures that the engine is equal to the python implementation in evaluation
If no exception is thrown, it has passed


'''

import os
import sys
import lcztools
from lcztools.testing.leela_engine import LCZeroEngine
import numpy as np
import chess.pgn
import time
import json
import collections


engine = LCZeroEngine()
board = lcztools.LeelaBoard()
# engine.evaluate(board())
net = lcztools.load_network()

def fix_policy_float(policy):
    '''Numpy to normal python float, for json dumps'''
    return collections.OrderedDict((k, float(v)) for k, v in policy.items())

def eval_equal(neteval, engineeval, tolerance=.00006):
    npol, nv = neteval
    epol, ev = engineeval
    for uci in npol:
        if abs(npol[uci] - epol[uci]) > tolerance:
            return False
    if (ev is not None) and (abs(nv - ev) > tolerance):
        return False
    return True

net_eval_time = 0
engine_eval_time = 0
totalevals = 0
numgames = 200
for gamenum in range(numgames):
    print("Playing game {}/{}".format(gamenum+1,numgames))
    board = lcztools.LeelaBoard()
    counter = 500
    while (counter>0) and (not board.is_game_over()):
        counter -= 1
        if board.can_claim_draw():
            counter = max(counter, 10)
        
        clock = time.time()
        policy, value = net.evaluate(board)
        net_eval_time += time.time() - clock
        
        clock = time.time()
        bestmove, epolicy, evalue = engine.evaluate(board)
        engine_eval_time += time.time() - clock
        
        totalevals += 1
        
        if not eval_equal((policy, value), (epolicy, evalue)):
            print("Note equal...")
            print("Policy:", json.dumps(fix_policy_float(policy), indent=3))
            print("Value:", value)
            print("Engine Bestmove:", bestmove)
            print("Engine Policy:", json.dumps(epolicy, indent=3))
            print("Engine Value:", evalue)
            raise Exception("Not equal:", ' '.join(m.uci() for m in board.move_stack))
        ucis = list(policy)
        pol_values = np.fromiter(policy.values(), dtype=np.float32)
        pol_values = pol_values/pol_values.sum()
        pol_index = np.random.choice(len(pol_values), p=pol_values)
        uci = ucis[pol_index]
        board.push_uci(uci)
    print()
    game = chess.pgn.Game.from_board(board.pc_board)
    print(game)
    print(board)
    print("Average net eval time   : {:.6f}".format(net_eval_time/totalevals))
    print("Average engine eval time: {:.6f}".format(engine_eval_time/totalevals))
    print("All Evals Equal: PASS")
print("Network looks good!")