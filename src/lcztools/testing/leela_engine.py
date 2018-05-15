# This module will interact with the Leela chess engine to get position evaluations
# Primary use case is threads=1 "go nodes 1" which is (usually) sufficient extract policy and value
# from info string... In the case of only one possible move, a depth 1 search will be done at that node,
# so it may not be possible to get the position's value (the root node's value is usually available
# at child-nodes with 0 visits)


import chess.uci
import chess
from collections import namedtuple, OrderedDict
import re
from operator import itemgetter
import sys



info_pattern = r' *(?P<san>\w*) ->' \
               r' *(?P<visits>\d*)' \
               r' *\(V: *(?P<value>[\d\.]*)\%\)' \
               r' *\(N: *(?P<policy>[\d\.]*)\%\)' \
               r' *PV: *(?P<pv>.*)$'

info_regex = re.compile(info_pattern)

InfoTuple = namedtuple('InfoTuple', 'san visits value policy pv')


class LCZInfoHandler(chess.uci.InfoHandler):
    def __init__(self):
        self.lcz_strings = []
        self.lcz_move_info = OrderedDict()
        super().__init__()
    def string(self, string):
        # Called whenever a complete info line has been processed.
        self.lcz_strings.append(string)
        match = info_regex.match(string)
        if match:
            info_tuple = InfoTuple(match.group('san'),
                                   int(match.group('visits')),
                                   float(match.group('value'))/100,
                                   float(match.group('policy'))/100,
                                   match.group('pv'))
            self.lcz_move_info[info_tuple.san] = info_tuple
        return super().string(string)
    def lcz_clear():
        self.lcz_strings.clear()
        self.lcz_move_info.clear()

class LCZEngine:
    def __init__(self, engine_path, weights_file, threads=1, visits=1, nodes=1, start=True):
        self.engine_path = engine_path
        self.weights_file = weights_file
        self.threads = threads
        self.visits = visits
        self.nodes = nodes
        self.info_handler = LCZInfoHandler()
        self.engine = None        
        if start:
            self.start()
    def start(self):
        command = [self.engine_path]
        command.extend(['-w', self.weights_file])
        if self.threads is not None:
            command.extend(['-t', self.threads])
        if self.visits is not None:
            command.extend(['-v', self.visits])
        command.extend(['-l', '/tmp/lczlog.txt'])            
        command = map(str, command)
        self.nodes = self.nodes
        self.engine = chess.uci.popen_engine(command)
        print("Leela engine started")
        self.engine.uci()
        self.engine.info_handlers.append(self.info_handler)
    def stop(self):
        self.engine.quit()
    def evaluate(self, board):
        '''returns a (policy, value) given a python-chess board where:
        policy is a mapping UCI=>value, sorted highest to lowest
        value is a float'''
        self.engine.position(board)
        self.engine.go(nodes=self.nodes)
        san_to_uci = {}
        value = None
        policy = {}
        for move in board.legal_moves:
            san = board.san(move)
            move_info = self.info_handler.lcz_move_info[san]
            if value is None and move_info.visits==0:
                value = move_info.value
            policy[move.uci()] = move_info.policy
        return OrderedDict(sorted(policy.items(), key=itemgetter(1), reverse=True)), value
          