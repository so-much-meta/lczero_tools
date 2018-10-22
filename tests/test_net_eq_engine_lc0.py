'''
This script just ensures that the engine is equal to the python implementation in evaluation
If no exception is thrown, it has passed


'''

import lcztools
from lcztools.testing.leela_engine_lc0 import LC0Engine
import numpy as np
import chess.pgn
import time
import json
import collections


# NOTE: Policy values seem to be a tiny bit off from lc0...
#       The reason for this seems to be usage of src/mcts/node.cc::Edge::SetP in lc0
#       This function applies some rounding to policy values
#
# Changing tolerance from 0.00006 to 0.0005
TOLERANCE =0.0005




engine = LC0Engine()
board = lcztools.LeelaBoard()
# engine.evaluate(board())
net = lcztools.load_network()

def fix_policy_float(policy):
    '''Numpy to normal python float, for json dumps'''
    return collections.OrderedDict((k, float(v)) for k, v in policy.items())


g_max_policy_error = 0
g_max_value_error = 0
g_mse_policy = 0
g_mse_value = 0
g_se_policy_sum = 0
g_se_value_sum = 0
g_policy_samples = 0
g_value_samples = 0

def eval_equal(neteval, engineeval, tolerance=TOLERANCE):
    global g_max_policy_error, g_max_value_error, g_mse_policy, g_mse_value, g_se_policy_sum, g_se_value_sum
    global g_policy_samples, g_value_samples
    npol, nv = neteval
    epol, ev = engineeval
    for uci in npol:
        policy_error = abs(npol[uci] - epol[uci])
        g_max_policy_error = max(policy_error, g_max_policy_error)
        g_policy_samples += 1
        g_se_policy_sum += policy_error**2
        if policy_error > tolerance:
            print("Policy not equal: {}, {}, {}".format(uci, npol[uci], epol[uci]))
            return False
    g_mse_policy = g_se_policy_sum / g_policy_samples
    value_error = abs(nv - ev)
    g_max_value_error = max(value_error, g_max_value_error)
    g_value_samples += 1
    g_se_value_sum += value_error**2
    g_mse_value = g_se_value_sum / g_value_samples
    if value_error > tolerance:
        print("Value not equal: {}, {}".format(nv, ev))
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
    engine.newgame()  # TODO -- It looks like without this, lc0 gives bad values.
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
            print("Not equal...")
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
    print("Max policy error: {:.7f}".format(g_max_policy_error))
    print("Max value error: {:.7f}".format(g_max_value_error))
    print("Policy MSE: {}".format(g_mse_policy))
    print("Value MSE: {}".format(g_mse_value))
    print("All Evals Equal: PASS")
print("Network looks good!")
