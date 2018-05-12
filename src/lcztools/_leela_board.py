import collections
import numpy as np
import chess
import struct
from lcztools._uci_to_idx import uci_to_idx as _uci_to_idx

flat_planes = []
for i in range(256):
    flat_planes.append(np.ones((8,8), dtype=np.float32)*i)

LeelaBoardData = collections.namedtuple('LeelaBoardData',
                            'white_planes black_planes rep_planes '
                            'transposition_key us_ooo us_oo them_ooo them_oo '
                            'side_to_move rule50_count')

class LeelaBoard(chess.Board):
    def __init__(self):
        super().__init__()
        self.lcz_stack = []
        self._lcz_transposition_counter = collections.Counter()
        self._lcz_push()        
    def _lcz_push(self):
        # Push data onto the lcz data stack after pushing board moves
        transposition_key = self._transposition_key()
        self._lcz_transposition_counter.update((transposition_key,))
        repetitions = self._lcz_transposition_counter[transposition_key] - 1
        # side_to_move = 0 if we're white, 1 if we're black
        side_to_move = 0 if self.turn else 1
        rule50_count = self.halfmove_clock
        # Figure out castling rights
        if not side_to_move:
            # we're white
            _c = self.castling_rights
            us_ooo, us_oo = (_c>>chess.A1) & 1, (_c>>chess.H1) & 1
            them_ooo, them_oo = (_c>>chess.A8) & 1, (_c>>chess.H8) & 1
        else: 
            # We're black
            _c = self.castling_rights
            us_ooo, us_oo = (_c>>chess.A8) & 1, (_c>>chess.H8) & 1
            them_ooo, them_oo = (_c>>chess.A1) & 1, (_c>>chess.H1) & 1
        # Create 13 planes... 6 us, 6 them, repetitions>=1
        white_planes = []
        black_planes = []
        for color, planes in ((True, white_planes), (False, black_planes)):
            for piece_type in range(1,7):
                byts = struct.pack('>Q', self.pieces_mask(piece_type, color))
                arr = np.unpackbits(bytearray(byts))[::-1].reshape(8,8).astype(np.float32)
                planes.append(arr)
        # planes.append(flat_planes[repetitions>=1])
        white_planes = np.stack(white_planes)
        black_planes = np.stack(black_planes)
        rep_planes = np.stack([flat_planes[repetitions>=1]])
        lcz_data = LeelaBoardData(
            white_planes=white_planes, black_planes=black_planes, rep_planes=rep_planes,
            transposition_key=transposition_key,
            us_ooo=us_ooo, us_oo=us_oo, them_ooo=them_ooo, them_oo=them_oo,
            side_to_move=side_to_move, rule50_count=rule50_count
        )
        self.lcz_stack.append(lcz_data)
    def push_uci(self, uci):
        super().push_uci(uci)
        self._lcz_push()
    def pop(self):
        result = super().pop()
        _lcz_data = self.lcz_stack.pop()
        self._transposition_counter.subtract((_lcz_data.transposition_key,))
        return result
    def features(self):
        '''Get neural network input planes'''
        # global planes
        planes = []
        curdata = self.lcz_stack[-1]
        for data in self.lcz_stack[-1:-9:-1]:
            if not curdata.side_to_move:
                # We're white
                planes.append(data.white_planes)
                planes.append(data.black_planes)
                planes.append(data.rep_planes)
            else:
                # We're black
                planes.append(data.black_planes[:,::-1])
                planes.append(data.white_planes[:,::-1])
                planes.append(data.rep_planes)
        planes = np.concatenate(planes)
        planes.resize(112,8,8)
        planes[-8] = curdata.us_ooo
        planes[-7] = curdata.us_oo
        planes[-6] = curdata.them_ooo
        planes[-5] = curdata.them_oo
        planes[-4] = curdata.side_to_move
        planes[-3] = curdata.rule50_count
        planes[-2] = 0
        planes[-1] = 1
        return planes
    def uci_to_idx(self, uci_list):
        # Return list of NN policy output indexes for this board position, given uci_list
        data = self.lcz_stack[-1]
        # uci_to_idx_index =
        #  White, no-castling => 0
        #  White, castling => 1
        #  Black, no-castling => 2
        #  Black, castling => 3
        uci_to_idx_index = (data.us_ooo | data.us_oo) +  2*data.side_to_move
        uci_idx_dct = _uci_to_idx[uci_to_idx_index]
        return [uci_idx_dct[m] for m in uci_list]
    def __repr__(self):
        return "LeelaBoard('{}')".format(self.fen())
    def __str__(self):
        boardstr = super().__str__() + \
                '\nTurn: {}'.format('White' if self.turn else 'Black')
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
