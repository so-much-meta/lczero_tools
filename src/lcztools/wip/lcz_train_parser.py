from collections import OrderedDict

class TrainingRecord:
    def __init__(self, data, player):
        self.data = data
        self.player = player

    def get_probabilities(self):
        '''Get all training probabilities for which are greater than 0, 
        sorted descending... returned as OrderedDict (uci => prob)'''
        # Global only for debug
        global pmoves

        offset = 4
        length = len(idx_to_move)*4 # 1858*4
        
        move_idx = (move_idx + num_moves)%num_moves
        pbytes = self.data[offset:offset + length]
        probs = struct.unpack('{}f'.format(len(idx_to_move)), pbytes)
        pmoves = [(uci, p) for uci, p in zip(idx_to_move, probs) if p>0]
        pmoves = sorted(pmoves, key=lambda mp: (-mp[1], mp[0]))
        
        if self.flip: # Flip black's move
            flipc = lambda nstr: str(ord('9') - ord(nstr))
            flip = lambda uci: uci[0]+flipc(uci[1])+uci[2]+flipc(uci[3])+uci[4:]
            pmoves = [(flip(uci), p) for uci, p in pmoves]
        return OrderedDict(pmoves)

    def get_piece_plane(self, history_index, current_player, piece_index):
        '''Given an 8-byte bit-plane, convert to uint8 numpy array
        history_index: number of moves ago; 0 means current board
        current_player: 0=white, 1=black
        piece_index: index of piece'''
        if player==0:
            return np.unpackbits(bytearray(bp)).reshape(8, 8)
        else:
            return np.unpackbits(bytearray(bp)).reshape(8, 8)[::-1]
    
    def get_move(self):
        '''Get UCI move, appropriately flipped'''
        
        # Check for K moves first bc of castling, pawns last for promotions...
        for idx, piece in reversed(indexed_pieces):
            arr1 = bp_to_array(planes1[idx], current_player==1)
            arr2 = bp_to_array(planes2[idx], current_player==0)
            if not np.array_equal(arr1, arr2):
                rowfrom, colfrom = np.where(arr1 & ~arr2)
                rowto, colto = np.where(~arr1 & arr2)
                promotion = ''
                if not len(colfrom)==len(rowfrom)==len(colto)==len(rowto)==1:
                    # This must be a pawn promotion...
                    assert (len(colfrom)==len(rowfrom)==0)
                    # Find where the pawn came from
                    p_arr1 = bp_to_array(planes1[0], current_player==1)
                    p_arr2 = bp_to_array(planes2[0], current_player==0)
                    rowfrom, colfrom = np.where(p_arr1 & ~p_arr2)
                    promotion = piece.lower()
                assert len(colfrom)==len(rowfrom)==len(colto)==len(rowto)==1
                rowfrom, colfrom = rowfrom[0], colfrom[0]
                rowto, colto = rowto[0], colto[0]
                uci = '{}{}{}{}{}'.format(columns[colfrom], rowfrom+1,
                                        columns[colto], rowto+1, promotion)
                return piece, uci
        else:
            raise Exception("I shouldn't be here")        
    

class TrainingData:
    '''Parse training data bytes'''
    RECORD_SIZE = 8276
    def __init__(self, data):
        s = self.RECORD_SIZE
        self.data = [data[i:i+s] for i in range(0,len(data),s)]

class TarTrainingFile:
    '''Parse training data'''
    def __init__(self, filename):
        self.filename = filename
        