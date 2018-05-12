# -*- coding: utf-8 -*-
import numpy as np


from collections import OrderedDict

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class LeelaNet:
    def __init__(self, model):
        self.model = model
    def evaluate(self, leela_board):
        features = leela_board.features()
        policy, value = self.model(features)
        if not isinstance(policy, np.ndarray):
            # Assume it's a torch tensor
            policy = policy.numpy()
            value = value.numpy()
        policy, value = policy[0], value[0][0]
#         if leela_board._board.turn:
#             idx_to_move_dict = dict((uci, idx) for idx, uci in enumerate(idx_to_move[0]))
#         else:
#             idx_to_move_dict = dict((uci, idx) for idx, uci in enumerate(idx_to_move[1]))
        legal_uci = [m.uci() for m in leela_board._board.generate_legal_moves()]
        legal_indexes = leela_board.uci_to_idx(legal_uci)
        # print(legal_uci)
        # print(policy)
        softmaxed = _softmax(policy[legal_indexes])
        policy_legal = OrderedDict(sorted(zip(legal_uci, softmaxed),
                                    key = lambda mp: (mp[1], mp[0]), 
                                    reverse=True))
        value = value/2 + 0.5
        return policy_legal, value
        

def load_network(backend, filename):
    backends = ('tensorflow', 'pytorch')
    if backend not in backends:
        raise Exception("Supported backends are {}".format(backends))
    if backend == 'tensorflow':
        from lcztools._leela_tf_net import LeelaLoader
    elif backend == 'pytorch':
        from lcztools._leela_torch_net import LeelaLoader
    return LeelaNet(LeelaLoader.from_weights_file(filename))
