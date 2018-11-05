import collections
import numpy as np
import chess
from chess import Move
import struct
from lcztools._uci_to_idx import uci_to_idx as _uci_to_idx
import zlib

flat_planes = []
for i in range(256):
    flat_planes.append(np.ones((8,8), dtype=np.uint8)*i)

LeelaBoardData = collections.namedtuple('LeelaBoardData',
                            'plane_bytes repetition '
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
    _plane_bytes_struct = struct.Struct('>Q')
    
    def __init__(self, leela_board = None, *args, **kwargs):
        '''If leela_board is passed as an argument, return a copy'''
        self.pc_board = chess.Board(*args, **kwargs)
        self.lcz_stack = []
        self._lcz_transposition_counter = collections.Counter()
        self._lcz_push()
        self.is_game_over = self.pc_method('is_game_over')
        self.can_claim_draw = self.pc_method('can_claim_draw')
        self.generate_legal_moves = self.pc_method('generate_legal_moves')
                
    def copy(self):
        """Note! Currently the copy constructor uses pc_board.copy(stack=False), which makes pops impossible"""
        cls = type(self)
        copied = cls.__new__(cls)
        copied.pc_board = self.pc_board.copy(stack=False)
        copied.pc_board.stack[:] = self.pc_board.stack[-7:]
        copied.pc_board.move_stack[:] = self.pc_board.move_stack[-7:]
        copied.lcz_stack = self.lcz_stack[:]
        copied._lcz_transposition_counter = self._lcz_transposition_counter.copy()
        copied.is_game_over = copied.pc_method('is_game_over')
        copied.can_claim_draw = copied.pc_method('can_claim_draw')
        copied.generate_legal_moves = copied.pc_method('generate_legal_moves')        
        return copied

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

    def push(self, move):
        self.pc_board.push(move)
        self._lcz_push()

    def push_uci(self, uci):
        # don't check for legality - it takes much longer to run...
        # self.pc_board.push_uci(uci)
        self.pc_board.push(Move.from_uci(uci))
        self._lcz_push()

    def push_san(self, san):
        self.pc_board.push_san(san)
        self._lcz_push()

    def pop(self):
        result = self.pc_board.pop()
        _lcz_data = self.lcz_stack.pop()
        self._lcz_transposition_counter.subtract((_lcz_data.transposition_key,))
        return result

    def _plane_bytes_iter(self):
        """Get plane bytes... used for _lcz_push"""
        pack = self._plane_bytes_struct.pack
        pieces_mask = self.pc_board.pieces_mask
        for color in (True, False):
            for piece_type in range(1,7):
                byts = pack(pieces_mask(piece_type, color))
                yield byts

    def _lcz_push(self):
        """Push data onto the lcz data stack after pushing board moves"""
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
        plane_bytes = b''.join(self._plane_bytes_iter())
        repetition = (repetitions>=1)
        lcz_data = LeelaBoardData(
            plane_bytes, repetition=repetition,
            transposition_key=transposition_key,
            us_ooo=us_ooo, us_oo=us_oo, them_ooo=them_ooo, them_oo=them_oo,
            side_to_move=side_to_move, rule50_count=rule50_count
        )
        self.lcz_stack.append(lcz_data)
        
    def serialize_features(self):
        '''Get compacted bytes representation of input planes'''
        planes = []
        curdata = self.lcz_stack[-1]
        bytes_false_true = bytes([False]), bytes([True])
        bytes_per_history = 97
        total_plane_bytes = bytes_per_history * 8
        def bytes_iter():
            plane_bytes_yielded = 0
            for data in self.lcz_stack[-1:-9:-1]:
                yield data.plane_bytes
                yield bytes_false_true[data.repetition]
                plane_bytes_yielded += bytes_per_history
            # 104 total piece planes... fill in missing with 0s
            yield bytes(total_plane_bytes - plane_bytes_yielded)
            # Yield the rest of the constant planes
            yield np.packbits((curdata.us_ooo,
                             curdata.us_oo,
                             curdata.them_ooo,
                             curdata.them_oo,
                             curdata.side_to_move)).tobytes()
            yield chr(curdata.rule50_count).encode()
        return b''.join(bytes_iter())
    
    @classmethod
    def deserialize_features(cls, serialized):
        planes_stack = []
        rule50_count = serialized[-1]  # last byte is rule 50
        board_attrs = np.unpackbits(bytearray(serialized[-2:-1]))  # second to last byte
        us_ooo, us_oo, them_ooo, them_oo, side_to_move = board_attrs[:5]
        bytes_per_history = 97
        for history_idx in range(0, bytes_per_history*8, bytes_per_history):
            plane_bytes = serialized[history_idx:history_idx+96]
            repetition = serialized[history_idx+96]
            if not side_to_move:
                # we're white
                planes = (np.unpackbits(bytearray(plane_bytes))[::-1]
                                        .reshape(12, 8, 8)[::-1])                
            else:
                # We're black
                planes = (np.unpackbits(bytearray(plane_bytes))[::-1]
                                        .reshape(12, 8, 8)[::-1]
                                        .reshape(2,6,8,8)[::-1,:,::-1]
                                        .reshape(12, 8,8))
            planes_stack.append(planes)
            planes_stack.append([flat_planes[repetition]])
        planes_stack.append([flat_planes[us_ooo],
            flat_planes[us_oo],
            flat_planes[them_ooo],
            flat_planes[them_oo],
            flat_planes[side_to_move],
            flat_planes[rule50_count],
            flat_planes[0],
            flat_planes[1]])     
        planes = np.concatenate(planes_stack)
        return planes                
        
    def lcz_features(self):
        '''Get neural network input planes as uint8'''
        # print(list(self._planes_iter()))
        planes_stack = []
        curdata = self.lcz_stack[-1]
        planes_yielded = 0
        for data in self.lcz_stack[-1:-9:-1]:
            plane_bytes = data.plane_bytes
            if not curdata.side_to_move:
                # we're white
                planes = (np.unpackbits(bytearray(plane_bytes))[::-1]
                                        .reshape(12, 8, 8)[::-1])                
            else:
                # We're black
                planes = (np.unpackbits(bytearray(plane_bytes))[::-1]
                                        .reshape(12, 8, 8)[::-1]
                                        .reshape(2,6,8,8)[::-1,:,::-1]
                                        .reshape(12, 8,8))
            planes_stack.append(planes)
            planes_stack.append([flat_planes[data.repetition]])
            planes_yielded += 13
        empty_planes = [flat_planes[0] for _ in range(104-planes_yielded)]
        if empty_planes:
            planes_stack.append(empty_planes)
        # Yield the rest of the constant planes
        planes_stack.append([flat_planes[curdata.us_ooo],
            flat_planes[curdata.us_oo],
            flat_planes[curdata.them_ooo],
            flat_planes[curdata.them_oo],
            flat_planes[curdata.side_to_move],
            flat_planes[curdata.rule50_count],
            flat_planes[0],
            flat_planes[1]])     
        planes = np.concatenate(planes_stack)
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
    
    def unicode(self):
        if self.pc_board.is_game_over() or self.is_draw():
            result = self.pc_board.result(claim_draw=True)
            turnstring = 'Result: {}'.format(result)
        else:
            turnstring = 'Turn: {}'.format('White' if self.pc_board.turn else 'Black') 
        boardstr = self.pc_board.unicode() + "\n" + turnstring
        return boardstr

    def __repr__(self):
        return "LeelaBoard('{}')".format(self.pc_board.fen())

    def _repr_svg_(self):
        return self.pc_board._repr_svg_()

    def __str__(self):
        if self.pc_board.is_game_over() or self.is_draw():
            result = self.pc_board.result(claim_draw=True)
            turnstring = 'Result: {}'.format(result)
        else:
            turnstring = 'Turn: {}'.format('White' if self.pc_board.turn else 'Black') 
        boardstr = self.pc_board.__str__() + "\n" + turnstring
        return boardstr
    
    def __eq__(self, other):
        return self.get_hash_key() == other.get_hash_key()
    
    def __hash__(self):
        return hash(self.get_hash_key())

    def get_hash_key(self):
        transposition_key = self.pc_board._transposition_key() 
        return (transposition_key +
                (self._lcz_transposition_counter[transposition_key], self.pc_board.halfmove_clock) +
                tuple(self.pc_board.move_stack[-7:])
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
