# -*- coding: utf-8 -*-
import numpy as np


from collections import OrderedDict

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class LeelaNet:
    def __init__(self, model):
        self.model = model
    def evaluate_batch(self, leela_boards):
        # TODO/Not implemented
        raise NotImplementedError
        features = []
        for board in leela_boards:
            features.append(board.features())
        features = np.stack(features)
        policy, value = self.model(features)
        if not isinstance(policy[0], np.ndarray):
            # Assume it's a torch tensor
            policy = policy.numpy()
            value = value.numpy()
        policy, value = policy[0], value[0][0]
        legal_uci = [m.uci() for m in leela_board.generate_legal_moves()]
        if legal_uci:
            legal_indexes = leela_board.lcz_uci_to_idx(legal_uci)
            softmaxed = _softmax(policy[legal_indexes])
            policy_legal = OrderedDict(sorted(zip(legal_uci, softmaxed),
                                        key = lambda mp: (mp[1], mp[0]),
                                        reverse=True))
        else:
            policy_legal = OrderedDict()
        value = value/2 + 0.5
        return policy_legal, value

    def evaluate(self, leela_board):
        features = leela_board.lcz_features()
        policy, value = self.model(features)
        if not isinstance(policy, np.ndarray):
            # Assume it's a torch tensor
            policy = policy.cpu().numpy()
            value = value.cpu().numpy()
        policy, value = policy[0], value[0][0]
        # Knight promotions are represented without a suffix in leela-chess
        # ==> the transformation is done in lcz_uci_to_idx
        legal_uci = [m.uci() for m in leela_board.generate_legal_moves()]
        if legal_uci:
            legal_indexes = leela_board.lcz_uci_to_idx(legal_uci)
            softmaxed = _softmax(policy[legal_indexes])
            policy_legal = OrderedDict(sorted(zip(legal_uci, softmaxed),
                                        key = lambda mp: (mp[1], mp[0]),
                                        reverse=True))
        else:
            policy_legal = OrderedDict()
        value = value/2 + 0.5
        return policy_legal, value


def load_network(backend, filename):
    backends = ('tensorflow', 'pytorch', 'pytorch_orig', 'pytorch_cuda')
    if backend not in backends:
        raise Exception("Supported backends are {}".format(backends))
    kwargs = {}
    if backend == 'tensorflow':
        from lcztools.backend._leela_tf_net import LeelaLoader
    elif backend == 'pytorch':
        from lcztools.backend._leela_torch_eval_net import LeelaLoader
    elif backend == 'pytorch_orig':
        from lcztools.backend._leela_torch_net import LeelaLoader
    elif backend == 'pytorch_cuda':
        from lcztools.backend._leela_torch_eval_net import LeelaLoader
        kwargs['cuda'] = True
    return LeelaNet(LeelaLoader.from_weights_file(filename, **kwargs))
