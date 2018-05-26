'''Parse train data according to:
https://github.com/glinscott/leela-chess/blob/master/training/tf/chunkparser.py#L115
'''
import numpy as np
import tarfile
from collections import OrderedDict
import struct
import chess
import chess.pgn
import os
from lcztools.util import tqdm, lazy_property

try:
    from lcztools._uci_to_idx import uci_to_idx as _uci_to_idx
    _idx_to_uci_wn = {v: k for k, v in _uci_to_idx[0].items()}
    _idx_to_uci_wc = {v: k for k, v in _uci_to_idx[1].items()}
    _idx_to_uci_bn = {v: k for k, v in _uci_to_idx[2].items()}
    _idx_to_uci_bc = {v: k for k, v in _uci_to_idx[3].items()}
    IDX_TO_UCI = [
        _idx_to_uci_wn, # White no castling
        _idx_to_uci_wc, # White castling
        _idx_to_uci_bn, # Black no castling
        _idx_to_uci_bc, # Black castling
    ]
except:
    print("No lcztools")


COLUMNS = 'abcdefgh'
INDEXED_PIECES = list(enumerate(['P', 'N', 'B', 'R', 'Q', 'K']))


class TrainingRecord:
    SCALARS_STRUCT = struct.Struct('<7B1b')
    PROBS_STRUCT = struct.Struct('<1858f')
    PIECES_STRUCT = struct.Struct('<6Q')
    def __init__(self, data):
        # s = struct.unpack_from('<I1858f', data)
        self.data = data
        self.version = data[0]
        if self.version != 3:
            raise Exception("Only version 3 of training data supported")
        (   self.us_ooo,
            self.us_oo,
            self.them_ooo,
            self.them_oo,
            self.side_to_move,
            self.rule50_count,
            self.move_count_unused,
            self.result) = TrainingRecord.SCALARS_STRUCT.unpack(data[-8:])
            
    def _get_last_moved_piece_index(self):
        '''Get the piece index of the last moved piece by opponent, K first'''
        piece_ints_t1 = TrainingRecord.PIECES_STRUCT.unpack(self.data[4 + 7432 + 6*8: 4 + 7432 + 12*8])
        piece_ints_t2 = TrainingRecord.PIECES_STRUCT.unpack(self.data[4 + 7432 + 19*8: 4 + 7432 + 25*8])
        for idx in range(5, -1, -1):
            p1 = piece_ints_t1[idx]
            p2 = piece_ints_t2[idx]
            if p1!=p2:
                return idx
        

    def get_probabilities(self):
        '''Get all moves with training probabilities greater than 0, 
        sorted descending... returned as OrderedDict (uci => prob)'''
        # Global only for debug
        # global pmoves
        probs = TrainingRecord.PROBS_STRUCT.unpack_from(self.data[4:])
        idx_to_uci_idx = 2*self.side_to_move + (self.us_ooo | self.us_oo)
        idx_to_uci = IDX_TO_UCI[idx_to_uci_idx]
        pmoves = []
        for idx, prob in enumerate(probs):
            if prob > 0:
                pmoves.append((idx_to_uci[idx], prob))
        pmoves = sorted(pmoves, key=lambda mp: (-mp[1], mp[0]))
        return OrderedDict(pmoves)

    def get_piece_plane(self, history_index, side_index, piece_index):
        '''Get a piece plane as an 8x8 numpy array, correctly flipped to normal (W-first) orientation
        
        history_index: range(0,8)
        side_index: 0 => our piece plane, 1 => their piece plane
        piece_index: index of piece'''
        # version_length + probs_length
        offset = 4 + 7432
        #  + len(INDEXED_PIECES) * num_sides * history_index
        offset += 8 * (6 * 2 + 1) * history_index
        # + len(INDEXED_PIECES) * side_index
        offset += 8 * 6 * side_index
        # + piece_index
        offset += 8 * piece_index
        
        planebytes = self.data[offset:offset+8]
        if self.side_to_move == 0:
            return np.unpackbits(bytearray(planebytes)).reshape(8, 8)
        else:
            return np.unpackbits(bytearray(planebytes)).reshape(8, 8)[::-1]
    
       
    

class TrainingGame:
    '''Parse training data bytes'''
    RECORD_SIZE = 8276
    
    def __init__(self, databytes, name):
        s = self.RECORD_SIZE
        self.records = [TrainingRecord(databytes[i:i+s]) for i in range(0, len(databytes),s)]
        self._cache = {}
        self.name = name
    
    def push_final_move(self, pc_board):
        '''Push the most likely final move, given python chess board and results in final record'''
        
        def test_final_move():
            if (result==1) and pc_board.is_checkmate():
                return True
            if (result==0) and (moveidx==449) and not pc_board.is_checkmate():
                # 450 move rule...
                return True
            if (result==0) and (pc_board.is_insufficient_material() or
                    pc_board.is_stalemate() or
                    pc_board.can_claim_draw()):
                return True
            return False
        
        record = self.records[-1]
        result = record.result
        if result == -1:
            print("Error -- Last position has score of -1?")
            return
        uci_probs = record.get_probabilities()
        promotion_save = []
        moveidx = len(pc_board.move_stack)
  
        for uci in uci_probs:
            try:
                pc_board.push_uci(uci)
            except ValueError:
                # This should be a knight promotion...
                # (But can be a queen promotion from older buggy engines)
                promotion_save.append(uci)
                uci = uci + 'n'
                pc_board.push_uci(uci)
            if test_final_move():
                break
            pc_board.pop()
        else:
            # Just in case we can't find the final move... We should never get here on
            # new data
            for uci in promotion_save:
                uci = uci + 'q'
                if uci not in uci_probs:
                    pc_board.push_uci(uci)
                    if test_final_move():
                        break
                    pc_board.pop()
            else:
                print("Error ({}) - no final move found!".format(self.name))

    def get_move(self, move_index):
        '''Get UCI move, appropriately flipped from piece plane comparison
        
        This will not get the final move
        Returns: (piece, UCI) tuple'''      
        record = self.records[move_index+1]
        piece_index = record._get_last_moved_piece_index()
        piece = INDEXED_PIECES[piece_index][1]
        
        arr1 = record.get_piece_plane(1, 1, piece_index)
        arr2 = record.get_piece_plane(0, 1, piece_index) 
        
        rowfrom, colfrom = np.where(arr1 & ~arr2)
        rowto, colto = np.where(~arr1 & arr2)
        
        promotion = ''
        if not len(colfrom)==len(rowfrom)==len(colto)==len(rowto)==1:
            # This must be a pawn promotion...
            assert (len(colfrom)==len(rowfrom)==0)
            # Find where the pawn came from
            p_arr1 = record.get_piece_plane(1, 1, 0)
            p_arr2 = record.get_piece_plane(0, 1, 0)
            rowfrom, colfrom = np.where(p_arr1 & ~p_arr2)
            promotion = piece.lower()
            assert len(colfrom)==len(rowfrom)==len(colto)==len(rowto)==1
        rowfrom, colfrom = rowfrom[0], colfrom[0]
        rowto, colto = rowto[0], colto[0]
        uci = '{}{}{}{}{}'.format(COLUMNS[colfrom], rowfrom+1,
                                COLUMNS[colto], rowto+1, promotion)
        return piece, uci

            
    def _get_move_orig(self, move_index):
        '''Get UCI move, appropriately flipped from piece plane comparison
        - This is a slower version of get_move()
        
        This will not get the final move
        Returns: (piece, UCI) tuple'''      
        record1 = self.records[move_index]
        record2 = self.records[move_index+1]
        for piece_index, piece in reversed(INDEXED_PIECES):
            arr1 = record1.get_piece_plane(0, 0, piece_index)
            arr2 = record2.get_piece_plane(0, 1, piece_index) 
            if not np.array_equal(arr1, arr2):
                arr1 = record1.get_piece_plane(0, 0, piece_index)
                arr2 = record2.get_piece_plane(0, 1, piece_index)                
                rowfrom, colfrom = np.where(arr1 & ~arr2)
                rowto, colto = np.where(~arr1 & arr2)
                promotion = ''
                if not len(colfrom)==len(rowfrom)==len(colto)==len(rowto)==1:
                    # This must be a pawn promotion...
                    assert (len(colfrom)==len(rowfrom)==0)
                    # Find where the pawn came from
                    p_arr1 = record1.get_piece_plane(0, 0, 0)
                    p_arr2 = record2.get_piece_plane(0, 1, 0)
                    rowfrom, colfrom = np.where(p_arr1 & ~p_arr2)
                    promotion = piece.lower()
                    assert len(colfrom)==len(rowfrom)==len(colto)==len(rowto)==1
                rowfrom, colfrom = rowfrom[0], colfrom[0]
                rowto, colto = rowto[0], colto[0]
                uci = '{}{}{}{}{}'.format(COLUMNS[colfrom], rowfrom+1,
                                        COLUMNS[colto], rowto+1, promotion)
                return piece, uci
        else:
            raise Exception("I shouldn't be here")
    
    def get_all_moves(self):
        # TODO: Need to be able to get final move...
        all_moves = [self.get_move(move_index) for move_index in range(len(self.records)-1)]
        return all_moves
    
    def get_pc_board(self, with_final_move = True):
        cache_key = ('get_pc_board', with_final_move) 
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        pc_board = chess.Board()
        for _piece, uci in self.get_all_moves():
            pc_board.push_uci(uci)
        if with_final_move:
            self.push_final_move(pc_board)
        self._cache[cache_key] = pc_board
        return pc_board
    
    def get_pgn(self, with_final_move = True):
        cache_key = ('get_pgn', with_final_move)
        if cache_key in self._cache:
            return str(self._cache[cache_key])
        pc_board = self.get_pc_board(with_final_move)
        pgn_game = chess.pgn.Game.from_board(pc_board)
        white_result = self.records[0].result
        pgn_game.headers["Event"] = self.name
        if white_result==1:
            pgn_game.headers["Result"] = "1-0"
        elif white_result==-1:
            pgn_game.headers["Result"] = "0-1"
        elif white_result==0:
            pgn_game.headers["Result"] = "1/2-1/2"
        else:
            print(white_result)
            raise Exception("Bad result")
        self._cache[cache_key] = pgn_game
        return str(pgn_game)
    

class TarTrainingFile:
    '''Parse training data'''
    def __init__(self, filename):
        self.filename = filename
        
    def __iter__(self):
        '''Generator to iterate through data'''
        def generator():
            with tarfile.open(self.filename) as f:
                for idx, member in enumerate(f):
                    databytes = f.extractfile(member).read()
                    yield TrainingGame(databytes, member.name)
        return generator()
    
    @lazy_property
    def archive_names(self):
        '''Read all the names from the training archive'''
        with tarfile.open(self.filename) as f:
            return f.getnames()
    
    def read_game(self, name):
        '''Read a single game from the archive'''
        name = str(name)
        names = self.archive_names
        with tarfile.open(self.filename) as f:
            # Search for any names that contain name
            names = [n for n in names if name in n]
            if len(names)==0:
                raise Exception("{} not found in {}".format(name, self.filename))
            elif len(names)>1:
                raise Exception("Multiple occurrences of {} found in {}".format(name, self.filename))
            databytes = f.extractfile(names[0]).read()
            return TrainingGame(databytes, names[0])
    
    def to_pgn(self, filename=None, progress=True):
        if progress:
            progress = tqdm
        else:
            progress = lambda it: it
        if filename is None:
            dirname = os.path.dirname(self.filename)
            basename = os.path.basename(self.filename)
            basename = basename.split('.')[0] + '.pgn'
            filename = os.path.join(dirname, basename)
        assert(os.path.abspath(self.filename) != os.path.abspath(filename))
        with open(filename, 'w') as pgn_file:
            for game in progress(self):
                pgn = game.get_pgn()
                pgn_file.write(pgn)
                pgn_file.write('\n\n\n')
                pgn_file.flush()
