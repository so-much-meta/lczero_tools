import zipfile
import chess.pgn
import io
import tqdm
import chess
import chess.uci
import re
import numpy as np
import os

try:
    engine.quit()
except:
    pass


if True:
    command = [os.path.expanduser('~/Downloads/stockfish-9-mac/Mac/stockfish-9-popcnt')]
    # command = ['/Applications/chessx.app/Contents/MacOS/data/engines-mac/uci/stockfish-8-64']
    engine = chess.uci.popen_engine(command)

engine.uci()

DEPTH = 9
ZEROS = [0]*DEPTH

def _info(self, arg):
    desc, score = regex.search(arg).groups()
    score = int(score)
    if desc == 'cp':
        self._v_cps[self._v_depth] = score
    elif desc == 'mate':
        self._v_mates[self._v_depth] = score
    else:
        raise Exception("oops")
    self._v_depth += 1    

stripfen = str.maketrans('', '', '0123456789/')

engine.__class__._info = _info
engine._v_cps = ZEROS.copy()
engine._v_mates = ZEROS.copy()
engine._v_depth = 0

regex = re.compile('score ([^ ]*) ([^ ]*) ')


columns = ['name', 'fen', 'turn', 'castling', 'en_passant', 'halfmove', 'fullmove', 'npieces']
columns += ['cp_{}'.format(n) for n in range(1, DEPTH+1)]
columns += ['mate_{}'.format(n) for n in range(1, DEPTH+1)]
columns += ['uci']
columns += ['finished', 'score', 'threefold', 'fifty', 'insufficient', 'stalemate']
all_records = []
def scoreit(game, name):
    global board
    board = chess.Board()
    records = []
    for idx, move in enumerate(game.main_line()):
        engine.position(board)
        engine._v_cps[:] = ZEROS
        engine._v_mates[:] = ZEROS
        engine._v_depth = 0
        engine.go(depth=DEPTH)
        fensplit = board.fen().split()
        npieces = len(fensplit[0].translate(stripfen))
        records.append([name] + fensplit + [npieces] + \
                       engine._v_cps + engine._v_mates + \
                       [move.uci()])
        board.push(move)
    if board.can_claim_threefold_repetition():
        finished = 1
        score = 0
        threefold = 1     
        fifty = 0
        insufficient = 0
        stalemate = 0
    elif board.can_claim_fifty_moves():
        finished = 1
        score = 0
        threefold = 0
        fifty = 1
        insufficient = 0        
        stalemate = 0
    elif board.is_insufficient_material():
        finished = 1
        score = 0
        threefold = 0
        fifty = 0
        insufficient = 1
        stalemate = 0
    elif board.is_stalemate():
        finished = 1
        score = 0
        threefold = 0
        fifty = 0
        insufficient = 0        
        stalemate = 1
    elif board.is_checkmate():
        finished = 1
        score = [1,0][board.turn]
        threefold = 0
        fifty = 0
        insufficient = 0        
        stalemate = 0        
    elif board.can_claim_draw():
        # Don't think this can happen...
        raise Exception("Why")
        finished = 1
        score = 0
        threefold = 0
        fifty = 0
        insufficient = 0        
        stalemate = 0
    elif not board.is_game_over():
        finished = 0
        score = 0
        threefold = 0
        fifty = 0
        insufficient = 0        
        stalemate = 0
    else:
        raise Exception("I don't know")
    result = [finished, score, threefold, fifty, insufficient, stalemate]
    for record in records:
        all_records.append(record + result)     


# with open('/Volumes/SeagateExternal/leela_data/match_games/LeelaMatchGamesPgn_200000-300000.zip', mode='r') as f:
with zipfile.ZipFile('/Volumes/SeagateExternal/leela_data/match_games/FixedLeelaMatchGamesPgn_200000-300000.zip') as z:
    namelist = z.namelist()
    for name in tqdm.tqdm(namelist[:20]):
        if name.endswith('.pgn'):
            game = chess.pgn.read_game(io.TextIOWrapper(z.open(name)))
            scoreit(game, name.split('/')[-1].split('.')[0])

