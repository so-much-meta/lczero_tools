# -*- coding: utf-8 -*-
import numpy as np

from lcztools.config import get_global_config
from collections import OrderedDict

def _softmax(x, softmax_temp):
    e_x = np.exp((x - np.max(x))/softmax_temp)
    return e_x / e_x.sum(axis=0)


class LeelaNet:
    def __init__(self, model, policy_softmax_temp = 1.0):
        self.model = model
        self.policy_softmax_temp = policy_softmax_temp
    
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
            softmaxed = _softmax(policy[legal_indexes], self.policy_softmax_temp)
            policy_legal = OrderedDict(sorted(zip(legal_uci, softmaxed),
                                        key = lambda mp: (mp[1], mp[0]),
                                        reverse=True))
        else:
            policy_legal = OrderedDict()
        value = value/2 + 0.5
        return policy_legal, value
    
    def evaluate_debug(self, leela_board, **kwargs):
        '''Same as evaluate, but allows debug kwargs.
        See LeelaBoard.lcz_features_debug'''
        features = leela_board.lcz_features_debug(**kwargs)
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
            softmaxed = _softmax(policy[legal_indexes], self.policy_softmax_temp)
            policy_legal = OrderedDict(sorted(zip(legal_uci, softmaxed),
                                        key = lambda mp: (mp[1], mp[0]),
                                        reverse=True))
        else:
            policy_legal = OrderedDict()
        value = value/2 + 0.5
        return policy_legal, value    

def list_backends():
    return ['pytorch_eval_cpu', 'pytorch_eval_cuda', 'pytorch_cpu', 'pytorch_cuda', 'tensorflow',
            'pytorch_train_cpu', 'pytorch_train_cuda']

def load_network(filename=None, backend=None, policy_softmax_temp=None):
    # Config will handle filename in read_weights_file
    config = get_global_config()
    backend = backend or config.backend
    policy_softmax_temp = policy_softmax_temp or config.policy_softmax_temp
    backends = list_backends()

    print("Loading network using backend={}, policy_softmax_temp={}".format(backend, policy_softmax_temp))
    if backend not in backends:
        raise Exception("Supported backends are {}".format(backends))

    kwargs = {}
    if backend == 'tensorflow':
        raise Exception("Tensorflow temporarily disabled, untested since latest changes")  # Temporarily
        from lcztools.backend._leela_tf_net import LeelaLoader
    elif backend == 'pytorch_eval_cpu':
        from lcztools.backend._leela_torch_eval_net import LeelaLoader
    elif backend == 'pytorch_eval_cuda':
        from lcztools.backend._leela_torch_eval_net import LeelaLoader
        kwargs['cuda'] = True
    elif backend == 'pytorch_cpu':
        from lcztools.backend._leela_torch_net import LeelaLoader
    elif backend == 'pytorch_cuda':
        from lcztools.backend._leela_torch_net import LeelaLoader
        kwargs['cuda'] = True
    elif backend == 'pytorch_train_cpu':
        from lcztools.backend._leela_torch_net import LeelaLoader
        kwargs['train'] = True
    elif backend == 'pytorch_train_cuda':
        from lcztools.backend._leela_torch_net import LeelaLoader
        kwargs['cuda'] = True
        kwargs['train'] = True
    return LeelaNet(LeelaLoader.from_weights_file(filename, **kwargs), policy_softmax_temp=policy_softmax_temp)
