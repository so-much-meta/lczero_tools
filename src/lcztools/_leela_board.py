import collections
import numpy as np
import chess
import struct
from lcztools._idx_to_move import idx_to_move

flat_planes = []
for i in range(256):
    flat_planes.append(np.ones((8,8), dtype=np.float32)*i)

LeelaBoardInfo = collections.namedtuple('LeelaBoardInfo',
                            'planes transposition_key repetitions us_ooo us_oo them_ooo them_oo '
                            'side_to_move rule50_count')

class LeelaBoard:
    def __init__(self):
        self._board = chess.Board()
        self._info_stack = []
        self._transposition_counter = collections.Counter()
        self._push_info()        
    def _push_info(self):
        transposition_key = self._board._transposition_key()
        self._transposition_counter.update((transposition_key,))
        repetitions = self._transposition_counter[transposition_key] - 1
        side_to_move = 0 if self._board.turn else 1
        rule50_count = self._board.halfmove_clock
        # Figure out castling rights
        _castle = struct.pack('>Q', self._board.castling_rights)
        _castle = np.unpackbits(bytearray(_castle))[::-1].reshape(8,8)
        if side_to_move:
            # we're black
            _castle = _castle[::-1]
        us_ooo, us_oo, them_ooo, them_oo = _castle[0,0], _castle[0,7], _castle[7,0], _castle[7, 7]
        # Create 13 planes... 6 us, 6 them, repetitions>=1
        planes = []
        for color in (self._board.turn, not self._board.turn):
            for piece_type in range(1,7):
                byts = struct.pack('>Q', self._board.pieces_mask(piece_type, color))
                arr = np.unpackbits(bytearray(byts))[::-1].reshape(8,8).astype(np.float32)
                if not self._board.turn:
                    # Flip if black
                    arr = arr[::-1]
                planes.append(arr)
        planes.append(flat_planes[repetitions>=1])
        planes = np.stack(planes)
        info = LeelaBoardInfo(
            planes=planes, transposition_key=transposition_key, repetitions=repetitions,
            us_ooo=us_ooo, us_oo=us_oo, them_ooo=them_ooo, them_oo=them_oo,
            side_to_move=side_to_move, rule50_count=rule50_count
        )
        self._info_stack.append(info)
    def push_uci(self, uci):
        self._board.push_uci(uci)
        self._push_info()
    def pop(self):
        self._board.pop()
        info = self._info_stack.pop()
        self._transposition_counter.subtract((info.transposition_key,))
    def features(self):
        '''Get neural network input planes'''
        # global planes
        planes = []
        flip = False
        for info in self._info_stack[-1:-9:-1]:
            if not flip:
                planes.append(info.planes)
            else:
                planes.append(info.planes[6:12,::-1,:])
                planes.append(info.planes[0:6,::-1,:])
                planes.append(info.planes[12:13,::,:])
            flip = not flip
        planes = np.concatenate(planes)
        planes.resize(112,8,8)
        info = self._info_stack[-1]
        planes[-8] = info.us_ooo
        planes[-7] = info.us_oo
        planes[-6] = info.them_ooo
        planes[-5] = info.them_oo
        planes[-4] = info.side_to_move
        planes[-3] = info.rule50_count
        planes[-2] = 0
        planes[-1] = 1
        return planes
    def _repr_svg_(self):
        return self._board._repr_svg_()
    def __repr__(self):
        return "LeelaBoard('{}')".format(self._board.fen())
    def __str__(self):
        boardstr = self._board.__str__() + \
                '\nTurn: {}'.format('White' if self._board.turn else 'Black')
        return boardstr

# lb = LeelaBoard()
# lb.push_uci('c2c4')
#lb.push_uci('c7c5')
#lb.push_uci('d2d3')
#lb.push_uci('c2c4')
#lb.push_uci('b8c6')
# saved_planes = planes
# planes = lb.features()
# output = leela_net(torch.from_numpy(planes).unsqueeze(0))
# output
