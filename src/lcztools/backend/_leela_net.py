# -*- coding: utf-8 -*-
import numpy as np

from lcztools.config import get_global_config
from collections import OrderedDict

def _softmax(x, softmax_temp):
    e_x = np.exp((x - np.max(x))/softmax_temp)
    return e_x / e_x.sum(axis=0)

class LeelaNetBase:
    def __init__(self, policy_softmax_temp = 1.0):
        self.policy_softmax_temp = policy_softmax_temp
    
    def call_model_eval(self, leela_board):
        """Get policy and value from model - this needs to be implemented in subclasses"""
        raise NotImplementedError
    
    def evaluate(self, leela_board):
        policy, value = self.call_model_eval(leela_board)
        return self._evaluate(leela_board, policy, value)
        
    def _evaluate(self, leela_board, policy, value):
        """This is separated from evaluate so that subclasses can evaluate based on raw policy/value"""
        if not isinstance(policy, np.ndarray):
            # Assume it's a torch tensor
            policy = policy.cpu().numpy()
            value = value.cpu().numpy()
        # Results must be converted to float because operations
        # on numpy scalars can be very slow
        value = float(value[0])
        # Knight promotions are represented without a suffix in leela-chess
        # ==> the transformation is done in lcz_uci_to_idx
        legal_uci = [m.uci() for m in leela_board.generate_legal_moves()]
        if legal_uci:
            legal_indexes = leela_board.lcz_uci_to_idx(legal_uci)
            softmaxed = _softmax(policy[legal_indexes], self.policy_softmax_temp)
            softmaxed_aspython = map(float, softmaxed)
            policy_legal = OrderedDict(sorted(zip(legal_uci, softmaxed_aspython),
                                        key = lambda mp: (mp[1], mp[0]),
                                        reverse=True))
        else:
            policy_legal = OrderedDict()
        value = value/2 + 0.5
        return policy_legal, value


class LeelaNet(LeelaNetBase):
    def __init__(self, model=None, policy_softmax_temp = 1.0, half=False):
        super().__init__(policy_softmax_temp=policy_softmax_temp)
        if half:
            self.dtype = np.float16
        else:
            self.dtype = np.float32        
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


    def call_model_eval(self, leela_board):
        features = leela_board.lcz_features()
        features = features.astype(self.dtype)
        policy, value = self.model(features)
        return policy[0], value[0]
        


def list_backends():
    return ['pytorch_eval_cpu', 'pytorch_eval_cuda', 'pytorch_cpu', 'pytorch_cuda', 'tensorflow',
            'pytorch_train_cpu', 'pytorch_train_cuda', 'net_client']

def load_network(filename=None, backend=None, policy_softmax_temp=None, network_id=None, half=None):
    # Config will handle filename in read_weights_file
    config = get_global_config()
    backend = backend or config.backend
    policy_softmax_temp = policy_softmax_temp or config.policy_softmax_temp
    backends = list_backends()

    print("Loading network using backend={}, policy_softmax_temp={}".format(backend, policy_softmax_temp))
    if backend not in backends:
        raise Exception("Supported backends are {}".format(backends))

    kwargs = {}
    if backend=='net_client':
        from lcztools.backend._leela_client_net import LeelaClientNet
        if filename != None:
            raise Exception('Weights file not allowed for net_client')
        if half is not None:
            print("Warning: half has no effect for LeelaClientNet -- this is done on server")
        return LeelaClientNet(policy_softmax_temp=policy_softmax_temp, network_id=network_id)
    
    half = half if half is not None else half
    if network_id != None:
        raise Exception("Network ID only for net_client backend")
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
    kwargs['half'] = half
    return LeelaNet(LeelaLoader.from_weights_file(filename, **kwargs), policy_softmax_temp=policy_softmax_temp, half=half)
