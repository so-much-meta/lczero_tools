# This module will interact with the Leela chess engine to get position evaluations
# Primary use case is threads=1 "go nodes 1" which is (usually) sufficient extract policy and value
# from info string... In the case of only one possible move, a depth 1 search will be done at that node,
# so it may not be possible to get the position's value (the root node's value is usually available
# at child-nodes with 0 visits)


import chess.uci
from lcztools import LeelaBoard
from collections import namedtuple, OrderedDict
import re
from operator import itemgetter
from lcztools.config import get_global_config
import os



info_pattern = (r'(?P<uci>\w*) +'  # The 
               r'\((?P<move_index>[\d]+) *\) +' # Move index
               r'N: *(?P<visits>\d+) +' # Visits
               r'\((?P<unused_1>[^\)]*)\) +' # Unknown/unused
               r'\(P: *(?P<policy>[\d\.]+)%\) +' # Policy percentage
               r'\(Q: *(?P<q_value>[\d\.-]+)\) +' # Q value
               r'\(U: *(?P<u_value>[\d\.-]+)\) +' # U value
               r'\(Q\+U: *(?P<q_u_value>[\d\.-]+)\) +' # Q+U value
               r'\(V: *(?P<value_str>[^ ]+)\)') # Value


## Example info strings/tests
# t1 = 'info string g2g4  (378 ) N:       0 (+ 0) (P:  2.50%) (Q: -1.03320) (U: 0.85851) (Q+U: -0.17469) (V:  -.----)'
# t1 = t1.replace('info string ', '')
# t2 = 'info string e2e4  (322 ) N:       2 (+ 8) (P: 14.04%) (Q:  0.04032) (U: 0.08678) (Q+U:  0.12710) (V:  0.0587)'
# t2 = t2.replace('info string ', '')


info_regex = re.compile(info_pattern)

InfoTuple = namedtuple('InfoTuple', 'uci move_index visits policy q_value u_value q_u_value value_str')


class LC0InfoHandler(chess.uci.InfoHandler):
    def __init__(self):
        self.lcz_strings = []
        self.lcz_move_info = OrderedDict()
        super().__init__()

    def string(self, string):
        # Called whenever a complete info line has been processed.
        self.lcz_strings.append(string)
        match = info_regex.match(string)
        if match:
            info_tuple = InfoTuple(match.group('uci'),
                  int(match.group('move_index')),
                  int(match.group('visits')),
                  float(match.group('policy'))/100,
                  float(match.group('q_value')),
                  float(match.group('u_value')),
                  float(match.group('q_u_value')),
                  match.group('value_str'))
            self.lcz_move_info[info_tuple.uci] = info_tuple
        return super().string(string)

    def lcz_clear(self):
        self.lcz_strings.clear()
        self.lcz_move_info.clear()


class LC0Engine:
    def __init__(self, engine_path=None, weights_file=None,
                 threads=1, nodes=1, backend='cudnn', start=True,
                 policy_softmax_temp=None,
                 logfile='lc0_log.txt', stderr='lc0.stderr.txt',
                 ):
        config = get_global_config()
        engine_path = engine_path or config.lc0_engine
        engine_path = os.path.expanduser(engine_path)
        self.policy_softmax_temp = policy_softmax_temp or config.policy_softmax_temp
        self.engine_path = engine_path 
        self.weights_file = config.get_weights_filename(weights_file)
        self.threads = threads
        self.nodes = nodes
        self.backend = backend
        self.info_handler = LC0InfoHandler()
        self.engine = None
        self.logfile = logfile
        self.stderrfile = stderr
        self.stderr = None
        if start:
            self.start()

    def start(self):
        print("lc0 outputting stderr to:", self.stderrfile)
        self.stderr = open(self.stderrfile, 'w')
        command = [self.engine_path, '--verbose-move-stats']
        weights_file = os.path.expanduser(self.weights_file)
        command.extend(['-w', weights_file])
        if self.threads is not None:
            command.extend(['-t', self.threads])
        if self.policy_softmax_temp is not None:
            command.extend(['--policy-softmax-temp={}'.format(self.policy_softmax_temp)])
        if self.logfile:
            print("lc0 logging to: {}".format(self.logfile))
            command.extend(['-l', self.logfile])
        command = list(map(str, command))
        self.engine = chess.uci.popen_engine(command, stderr=self.stderr)
        print("Leela lc0 engine started with command: {}".format(' '.join(command)))
        self.engine.info_handlers.append(self.info_handler)
        self.engine.uci()


    def stop(self):
        self.engine.quit()
        try:
            self.stderr.close()
        except:
            pass

    def newgame(self):
        self.engine.ucinewgame()

    def evaluate(self, board):
        '''returns a (bestmove, policy, value) tuple given a python-chess board where:
        policy is a mapping UCI=>value, sorted highest to lowest
        value is a float'''
        if isinstance(board, LeelaBoard):
            board = board.pc_board
        
        # Note/Important - this 'is_reversible' fix is needed to make sure that the engine
        # gets all history when calling self.engine.position(board)
        board = board.copy()
        board.is_irreversible = lambda move: False
            
        self.info_handler.lcz_clear()
        self.engine.position(board)
        bestmove = self.engine.go(nodes=self.nodes)
        value = None
        policy = {}
        for move in board.legal_moves:
            uci = board.uci(move)
            move_info = self.info_handler.lcz_move_info[uci]
            if value is None and move_info.visits==0:
                value = (move_info.q_value + 1)/2
            policy[uci] = move_info.policy
        return bestmove, OrderedDict(sorted(policy.items(), key=itemgetter(1), reverse=True)), value
