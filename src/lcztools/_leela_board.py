import collections
import numpy as np
import chess
import struct
from lcztools._uci_to_idx import uci_to_idx as _uci_to_idx
import zlib

flat_planes = []
for i in range(256):
    flat_planes.append(np.ones((8,8), dtype=np.float32)*i)

LeelaBoardData = collections.namedtuple('LeelaBoardData',
                            'white_planes black_planes rep_planes '
                            'transposition_key us_ooo us_oo them_ooo them_oo '
                            'side_to_move rule50_count')

def pc_board_property(propertyname):
    '''Create a property based on self.pc_board'''
    def prop(self):
        return getattr(self.pc_board, propertyname)
    return property(prop)
    
class LeelaBoard:
    turn = pc_board_property('turn')
    move_stack = pc_board_property('move_stack')
    
    def __init__(self, leela_board = None, *args, **kwargs):
        '''If leela_board is passed as an argument, return a copy'''
        if leela_board:
            # Copy
            self.pc_board = leela_board.pc_board.copy(stack=False)
            self.lcz_stack = leela_board.lcz_stack[:]
            self._lcz_transposition_counter = leela_board._lcz_transposition_counter.copy()
        else:
            self.pc_board = chess.Board(*args, **kwargs)
            self.lcz_stack = []
            self._lcz_transposition_counter = collections.Counter()
            self._lcz_push()
        self.is_game_over = self.pc_method('is_game_over')
        self.can_claim_draw = self.pc_method('can_claim_draw')
        self.generate_legal_moves = self.pc_method('generate_legal_moves')
                
    def copy(self):
        """Note! Currently the copy constructor uses pc_board.copy(stack=False), which makes pops impossible"""
        return self.__class__(leela_board=self)

    def pc_method(self, methodname):
        '''Return attribute of self.pc_board, useful for copying method bindings'''
        return getattr(self.pc_board, methodname)
    
    def is_threefold(self):
        transposition_key = self.pc_board._transposition_key()
        return self._lcz_transposition_counter[transposition_key] >= 3
    
    def is_fifty_moves(self):
        return self.pc_board.halfmove_clock >= 100
    
    def is_draw(self):
        return self.is_threefold() or self.is_fifty_moves()

    def _lcz_push(self):
        # print("_lcz_push")
        # Push data onto the lcz data stack after pushing board moves
        transposition_key = self.pc_board._transposition_key()
        self._lcz_transposition_counter.update((transposition_key,))
        repetitions = self._lcz_transposition_counter[transposition_key] - 1
        # side_to_move = 0 if we're white, 1 if we're black
        side_to_move = 0 if self.pc_board.turn else 1
        rule50_count = self.pc_board.halfmove_clock
        # Figure out castling rights
        if not side_to_move:
            # we're white
            _c = self.pc_board.castling_rights
            us_ooo, us_oo = (_c>>chess.A1) & 1, (_c>>chess.H1) & 1
            them_ooo, them_oo = (_c>>chess.A8) & 1, (_c>>chess.H8) & 1
        else: 
            # We're black
            _c = self.pc_board.castling_rights
            us_ooo, us_oo = (_c>>chess.A8) & 1, (_c>>chess.H8) & 1
            them_ooo, them_oo = (_c>>chess.A1) & 1, (_c>>chess.H1) & 1
        # Create 13 planes... 6 us, 6 them, repetitions>=1
        white_planes = []
        black_planes = []
        for color, planes in ((True, white_planes), (False, black_planes)):
            for piece_type in range(1,7):
                byts = struct.pack('>Q', self.pc_board.pieces_mask(piece_type, color))
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

    def push(self, move):
        self.pc_board.push(move)
        self._lcz_push()

    def push_uci(self, uci):
        self.pc_board.push_uci(uci)
        self._lcz_push()

    def push_san(self, san):
        self.pc_board.push_san(san)
        self._lcz_push()

    def pop(self):
        result = self.pc_board.pop()
        _lcz_data = self.lcz_stack.pop()
        self._lcz_transposition_counter.subtract((_lcz_data.transposition_key,))
        return result

    def lcz_features(self):
        '''Get neural network input planes'''
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
        planes.resize((112,8,8), refcheck=False)
        planes[-8] = curdata.us_ooo
        planes[-7] = curdata.us_oo
        planes[-6] = curdata.them_ooo
        planes[-5] = curdata.them_oo
        planes[-4] = curdata.side_to_move
        planes[-3] = curdata.rule50_count
        planes[-2] = 0
        planes[-1] = 1
        return planes

    def lcz_features_debug(self, fake_history=False, no_history=False, real_history=7, rule50=None, allones=None):
        '''Get neural network input planes, with ability to modify based on parameters'''
        planes = []
        curdata = self.lcz_stack[-1]
        if no_history:
            real_history = 0
        num_filled = 0
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
            num_filled +=1
            real_history -= 1
            if real_history<0:
                break
        # Augment with fake history, reusing last data
        if fake_history:
            for _ in range(8 - num_filled):
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
        if rule50 is not None:
            planes[-3] = rule50
        else:
            planes[-3] = curdata.rule50_count
        planes[-2] = 0
        if allones is not None:
            planes[-1] = allones
        else:
            planes[-1] = 1
        return planes

    def lcz_uci_to_idx(self, uci_list):
        # Return list of NN policy output indexes for this board position, given uci_list
        
        # TODO: Perhaps it's possible to just add the uci knight promotion move to the index dict
        # currently knight promotions are not in the dict 
        uci_list = [uci.rstrip('n') for uci in uci_list]
        
        data = self.lcz_stack[-1]
        # uci_to_idx_index =
        #  White, no-castling => 0
        #  White, castling => 1
        #  Black, no-castling => 2
        #  Black, castling => 3
        uci_to_idx_index = (data.us_ooo | data.us_oo) +  2*data.side_to_move
        uci_idx_dct = _uci_to_idx[uci_to_idx_index]
        return [uci_idx_dct[m] for m in uci_list]
    
    @classmethod
    def compress_features(cls, features):
        """Compress a features array as returned from lcz_features method"""
        features_8 = features.astype(np.uint8)
        # Simple compression would do this...
        # return zlib.compress(features_8)
        piece_plane_bytes = np.packbits(features_8[:-8]).tobytes()
        scalar_bytes = features_8[-8:][:,0,0].tobytes()
        compressed = zlib.compress(piece_plane_bytes + scalar_bytes)
        return compressed
    
    @classmethod
    def decompress_features(cls, compressed_features):
        """Decompress a compressed features array from compress_features"""
        decompressed = zlib.decompress(compressed_features)
        # Simple decompression would do this
        # return np.frombuffer(decompressed, dtype=np.uint8).astype(np.float32).reshape(-1,8,8)
        piece_plane_bytes = decompressed[:-8]
        scalar_bytes = decompressed[-8:]
        piece_plane_arr = np.unpackbits(bytearray(piece_plane_bytes))
        scalar_arr = np.frombuffer(scalar_bytes, dtype=np.uint8).repeat(64)
        result = np.concatenate((piece_plane_arr, scalar_arr)).astype(np.float32).reshape(-1,8,8)
        return result    
    

    def __repr__(self):
        return "LeelaBoard('{}')".format(self.pc_board.fen())

    def _repr_svg_(self):
        return self.pc_board._repr_svg_()

    def __str__(self):
        boardstr = self.pc_board.__str__() + \
                '\nTurn: {}'.format('White' if self.pc_board.turn else 'Black')
        return boardstr
    
    def __eq__(self, other):
        return self.get_hash_key() == other.get_hash_key()
    
    def __hash__(self):
        return hash(self.get_hash_key())

    def get_hash_key(self):
        transposition_key = self.pc_board._transposition_key() 
        return (transposition_key +
                (self._lcz_transposition_counter[transposition_key], self.pc_board.halfmove_clock) +
                tuple(self.pc_board.move_stack[-8:])
                )

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
