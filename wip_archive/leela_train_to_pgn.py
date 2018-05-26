'''
Get PGN data from Leela training data, according to format at:
https://github.com/glinscott/leela-chess/blob/master/training/tf/chunkparser.py#L115
'''
import tarfile
import os
import numpy as np
import chess
from collections import namedtuple, defaultdict
from chess import pgn
import zlib, base64
import struct

# filename = './games7580000.tar.gz'

filename = './games9030000.tar.gz'
outputdir = './pgn'
ensure_legal = False # Check for move legality (requires generating moves - slow)
include_header = True # include header in PGN output?
single_file = True # output to one file
global_stats = False # Create big dictionary of all possible moves, and how often they are legal/played
add_final_move = True # Add final move, based on training move probabilities and W/L/D result



# This is just a zlib-compressed -> base85 representation of:
# https://github.com/glinscott/leela-chess/blob/master/lc0/src/chess/bitboard.cc#L26
# To decode...
#    import zlib, base64
#    zlib.decompress(base64.b85decode(idx_to_move.strip().replace('\n', ''))).decode().split()
idx_to_move = '''
c-ke|Np|bV4h6uiye)h+B)8*i08H)u4?ZGma*u4Pbi>RfROI{l-}v9ZpZ`s<#1<h^l=yjd3^B#e+mB+3pO61m{Cs_Y__=(F__=+Q
_<4NZ^Kbm`=YJhjEU`t16eYTHs2nQ$eWH7Z-l4aDcYgf%05QHkMU2ZwiE;ZpF`j=@@AUUhy;JYhJM~VzQ}5I}^-jJ0KGEe<`BXlY
_kr`{$ES$-^-*G8K2OYBJpY#RrF{9zm-3~2DPPK$-lccxU3&X1qI;L#rFZG=1ESCUa_;+6&a1DF66+GTc>ZnWTlrSL{rOw@R=$;Q
y<6|ryY==_(Y;&m*1Prg0nz7vJNNwoqE~-g{f<lA;`tZKLwP6<<>Ai{<)J+EhThN{di#Lr-q0I*LvJ4tecHom4+|d_KJ2=%>pE`n
{7dDjJe8;NRG$9)RGxZMZ|Y6GsWMfj%2b)ob2`syrPE5My_WV`+URMccRc?}Z|N<)rML8!-tw21%2HV>OIv9xZRNz46I)ibtZLa;
WnYz@U3PZ4tjcBe^Xkf<m&^Gz&#RYLVu>w6q$uG^ez}rguH=_1`8C88F6Y-0F6Y-4F6UPWm-DOt)S-9i9eT&hHT`n^Dj&*+@}Ybv
AIgXFp?oOs_lXt^f<e%~J3oGXfN){ILb$MBDF#9R`4eMeOpJ*!UoLT$>1<4v$ue0c%Ve1>lV!3@mVTdT&;*)56Q~cIAHP0DxY%DQ
T<otBu6UNkvRD?&db#3(7SM8r7SIA(KnrNGESANxSo$rZEsJHbES5eXdbO6->Q6bZE*~XKftM-3vRO9EX4x+T0B8ekpzSnmpbfNv
Hp^z&ESsf|ineT)&9YhgfarDG)~!E4^d8#wP{%ELGOz?oumnrI%mbhR3ZMWArzwB}C|H6eSc0Vwh_(bvumnpV5WQq!$-+ho8!7DP
u%A16TCgNbvLs8gBula+OR}Uhlq`vn7>SV>Nst6dkOXOY((<HDlQvD-%4sX7D=J-4MNkAqPy|I#1VvB;MNm#jDKF)vy!4jd(py%c
tU}oqWm}ZZS~hFBgvurK^EN@=Cdk_adHddf8zb*|UcJ2%OKcG$MF|t+ZGyZ_khcl)9%2e(<h_J3^4`K2dH1;=%7^lye7ubjf<Z6{
`gg+Iz!(?<V_*!7fiW-!#=z+JiM9-u!P37wKVCjS7%1-$2FklvVFFE{2{eJ`+d!e3RFi5_^#NgwI8766qD{1kHqj>9M4M<6t=}iw
H~A*t<m&_H$IGV(^W~jl^7WD}zQwor7T<asGJK0~@h!eSAR4(Kmy@_47vzFmkc)5eExyIqZxQWVe2Z`K^#RdKyDV*g%6WDBC}HsM
ZNAO7`S#o3;oE$hZ}as5(Z~(CA-B`HAvffP+<cpF^KHI9D%!XCHs9v!1EQCETkie<(VKAFgdIJM_<}F^f-l~N5nu2HU-0z-(a3-d
$bbx|Gav&p_<}F^g0ByV_61+?1z#T!z1(5B!=4O#GF&9#B1yjFOTOewv_wm^L`$?pOSD8wv~&^^Evb?!sgf#z5-5QZD6MK*)wKW8
{!150x=4zpSc;`E3ZpOzqc94iFbbnE%4sTsA}E4VUdl^(S*@~KWlNPURW^Uw{68P_>f^TMV^n>Ns*h3i@jo^n^XfCttB+U0eadHx
5GhI+RUf13V^n>Ns?QKpm{*@A%&SlDm+>*GKJHzV59LGo_?S`zgJ2N!?}V|0F)#+kz!(?<V_*!7fzj_1Z5b?srGIyRynTQ$uRgsu
=Es!!xSs)<Koe*J&5r>^HK``mr0N60SaO;s+C-aZ6K$eRw23y+CR)Ewv~TiFzRA}I&X2cG5$4sW_sg<m%aW~+`yR*zxgZzh`j}FX
3vxj&$UY$4@{lfPa*;05MY>2A<bqs~3$ouL8o3}B<bv!2qStp>-~N>IinMpww#eHe?~j`#(oMQaH|hSESEQSClWx*JAey^zH|}<J
H}1yWxSMp7ZqiNKM@36F=_cKzeL(c;Z>!%QAbv>AFk2vOfq)FifDFj^7+R148IS?l2SiJQG)RLqoZTP|(tr%efDFh!AQ~Bv0U3~e
K=c*}TOjPyuus$Kr`1os<V(KfOSD8wv_wm^L`$?pOSE(*6D_HdDyfnxff6Wz5-2TlTI94R)1EBrTh_N&iltZzqc94iFbbnE3ZpOz
qnxH9D1ss=<)yromnADpR(4p~VLx9p?CYt^*9`l*SNj@YU*qd*e0}|2)z=LB&hzT)m2h+R^}yvDB1H+~>uY>{jjylq^&MggGwj=f
@ioJ~9=&{xudmxQ<wN;UKE5Uw!5|m}{X1b)VGN9cF)#+kz!(?<V_@|AL|X>SVCmnTA6a?>&#xKw^*H8ha(&&V0ZpI@G=b*V(4v}D
lWJ1+0bx`*O%rXRO|*$N(I(nNn`jfQ-zVBP`6l1w>jURUzTUv=YkYk@v9V;!lC7^>H^>FKAQ$BN8d{JGazQT0J|Nt+kuGO)kuK6j
x=0t~f?SXbvfm;axgZzhg6sq0=O9II?`?-|J8WCzZISoaJss&L-K3jze+@0tO}a@pX&(@t@!)RU?d)#cjk|F-=_cKzo3xLLmTuBb
x=H(h7?4}^1rhdS*pp!kge?$|0U3}18DB#SG9UvoAp3x5X^;kKkcP7xq(K^x0U3}1*#|@;12P~3vJZ&KXNK9QX`iOmPphAN$(MY|
muQKWXo;3+iI!-ImT2irCR$P@RZ=BY0wquaB~V)Aw8&*omOWY4x2$im6icxbMqw02VH8GT6h>hbMmbGIPy|I#%1e1EFH2UI?B_DW
E)R__GwkxH=`zDEH;R|>bs1lm@pbt>t;-C%=6Q8_CEOWao-$pY9A7C)7+;t1bs1lm@pTQ+>oG1f?DFLJGQ%!UnJ(k&a%ZS~C?Cql
WpWV=f<e%~6Gj!rz!(?<V_*!7fiW-!M!!$|9Dz8NUXOX1VV9@Lml<|>%5<4rms>%g2{eHw(7X&Ss!27eCRHC0MwQbv(I(nNn`jel
qD{1kHqrWh;s<CPl{mg$w{@9em&tW`-ek3w)moRkK)%Jd_!i%~%qzadxA+!c9}sQ;A(xZ5AQ$9<T#$=z@h!f^*KZL6Xdaa~zP`%#
WoX$_+m_l^bz9YaxdG(ce4B6c?aP$n+kBgE^YsDY85HD(+)n3)+>je`^KHJ(xB2?0m_X~O#PRi&7dC&`{9*rv{TG%yEO+n)U+@KA
Tm}?h@C9G+^#S22He^5sWH_Ax8IZvje8CrdeLyTA|0l>7|M~g~O&3YJNYb86donF|TJGdazT`{3L`$?pOSD8wv_wm^L`x?z(UK~u
k}9bZD1j0vfyzZvE|Rhd%O)&KTb8z1iltZzqc94iFbbnE3ZpOzqnxH9D1ss=<)yro|J>%)?ccX<^Xm3g{5G#{&$Mpy>UN`e8&$Va
bsJT;|Es#qt9zbTw^zcQ;q7tO?a}z{pS$i7M%8Up-NwUhRNel4>o%%x&%$q`>h}EVHl=Pig?fkHp?BN{l=7i`DDU416Nz9D41z&0
2nN9*7zE=#fdOzEP`BB1d-#2uO}8ghw~2JS?IV~3lVB3e+c3hI7!zY+^Z{W6IVF>2vP_oAGFc|eWSK1UKZ}d~HjHk==r)3G&!DWr
vI^^VW5=>s7RzE;x52}*SQg7-=>x(&9nf-y7SIA(KnrNGESANx{<FBSZ-eJHb8L%kTWrg-EziE)wXtlL&9YhcZOE`}md&zR`hf7f
31|atr)dLipbfNHHp^z&|5;qbxA|hThRqr_P1rPH$-<HaORxk>u*7Ykumnr61WO+f9)1G_PymI~6hHwKEWr{i@t?&&aS5eMC~f7m
mD5H_8!4?@TDJsBpae>wWJ#7}NtR?umSjnmWJzZzSrQ{L5+gCn6;-aNvY*R-E_<l#p|V<KwThq!ilCI2@={*POL-|T<)yrom$OlN
OK<7@c}$SUKaM{p$m5^X9uws8K>RU59#5Gb6XbE5_81<I;qe$AkN>B6Ops@uS3Y!#CAJ8W!oQk7hR0)WJpTFYF*hEMydQJp@f_(f
G9EW#+J?5FZ9Jxh%As<o?B5CFLGREz^bWm4@6bCR)8a8L9)sfX6#FqK9*>3|<Kb~{rE;pADyPc%m<)QS-l=!$?E}I*@Rv{JQ~6Xr
l~3jCF&Q3{;V~H=^WgD}$BA7|?0VcO>0Nr4-lcavrhwk1cj;Yv`+#thq<s1Fm-3~2DPPL>V+uT`z+(Vd)ooR`^Sqts{kY}PyY+6p
Tkn2c@p`x3t#|9~1Huy<<y-mo=Wpd(`Bol}EB<k@+gD*<g_RB~9Zq{V?V&gHhThN{kE>g6=ncK0w+{$Ula+_^P#*sLP#((jaberp
X=kUsmiAg&__Xlp+^2J&%2RnNPvxmM^`_p`n|f1k>P@|=H~r<Qw_H}`vML+BZ1l40%C0M`Usk`I`*Q9}Z|N<)rLt6(%2HV>OJ%7n
m8G)IwbfrrzbF4rj`Mx}d-Ct(IA5Z_C;v{4^Nsp@^6%t0U$4I>|4xqcP4|29@8mdNd%q|DPLA^(_<Qp2<TzhPzbF4rj`Q90d-Ct(
IA2u1C;v{4^PTZ~^6%t0Um(9H|4xqcZSs5a@8mdx{rBYG$#JIq@5#TD<Ba^@lYb}28Nt6N|4xoGkAF}8og8O4|DOCiInM0-J^6QX
oI(0~^6%vMKet}91^
'''

idx_to_move = zlib.decompress(base64.b85decode(idx_to_move.strip().replace('\n', ''))).decode().split()

class MoveStat:
    def __init__(self):
        self.legal_count = 0
        self.play_count = 0
    def __repr__(self):
        return('MoveStat(legal_count={}, play_count={})'
               .format(self.legal_count, self.play_count))

        
if global_stats:
    ensure_legal = True
    move_stats = defaultdict(MoveStat)

try:
    os.makedirs(outputdir)
except OSError as e:
    if not os.path.exists(outputdir):
        raise

indexed_pieces = list(enumerate(['P', 'N', 'B', 'R', 'Q', 'K']))
columns = 'abcdefgh'

def get_training_probabilities(data, move_idx):
    '''Get all training probabilities for move_idx, which are greater than 0, 
    sorted descending... returned as list of (uci, prob) tuples'''
    # Global only for debug
    global pmoves

    chunksize = 8276
    offset = 4
    length = len(idx_to_move)*4 # 1858*4
    
    num_moves = len(data)//chunksize
    move_idx = (move_idx + num_moves)%num_moves
    pbytes = data[move_idx*chunksize + offset:
                  move_idx*chunksize + offset + length]
    probs = struct.unpack('{}f'.format(len(idx_to_move)), pbytes)
    pmoves = [(uci, p) for uci, p in zip(idx_to_move, probs) if p>0]
    pmoves = sorted(pmoves, key=lambda mp: (-mp[1], mp[0]))
    
    if move_idx%2 == 1: # Flip black's move
        flipc = lambda nstr: str(ord('9') - ord(nstr))
        flip = lambda uci: uci[0]+flipc(uci[1])+uci[2]+flipc(uci[3])+uci[4:]
        pmoves = [(flip(uci), p) for uci, p in pmoves]
    # print(pmoves)
    return pmoves    

def get_sorted_final_moves(data):
    '''For final move..
    Return a list of (UCI move, probability) tuples, sorted descending based on probs
    '''
    return get_training_probabilities(data, -1)
    

def getbps_result(data):
    '''Get my-move bit-planes:
        returns an array (number of move elements) of arrays (each with 2 elements, 1 per side)
            of arrays (1 per piece, 6 total) of 8-bytes'''
    chunksize = 8276
    offset = 4+7432
    numplanes = 6
    bps = []
    assert len(data)%chunksize == 0
    for i in range(len(data)//chunksize):
        sidebps = []
        for side in range(2):
            planes = data[i*chunksize+offset + side*numplanes*8
                          :i*chunksize + offset + numplanes*8 + side*numplanes*8]
            piecebbs = []
            for plane in range(numplanes):
                piecebbs.append(planes[plane*8:(plane+1)*8])
            sidebps.append(piecebbs)
        bps.append(sidebps)
    result_offset = chunksize - 1
    result = np.int8(data[i*chunksize + result_offset])
    if i%2==0:
        # I am white
        white_result = result
    else:
        white_result = -result
    return bps, white_result

def bp_to_array(bp, flip):
    '''Given an 8-byte bit-plane, convert to uint8 numpy array'''
    if not flip:
        return np.unpackbits(bytearray(bp)).reshape(8, 8)
    else:
        return np.unpackbits(bytearray(bp)).reshape(8, 8)[::-1]


def convert_to_move(planes1, planes2, move_index):
    '''Given two arrays of 8-byte bit-planes, convert to a move
    Also need the index of the first move (0-indexed) to determine how to flip the board.
    '''
    current_player = move_index % 2
    
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

def getpgn(data, name):
    # These are global just for debug
    global game, node, white_result, legal_moves, board, uci, move, moveidx, final_moves, final_move_probs
    game = chess.pgn.Game()
    game.headers["Event"] = name
    node = game
    bps, white_result = getbps_result(data)
    if white_result==1:
        game.headers["Result"] = "1-0"
    elif white_result==-1:
        game.headers["Result"] = "0-1"
    elif white_result==0:
        game.headers["Result"] = "1/2-1/2"
    else:
        print(white_result)
        raise Exception("Bad result")
    for moveidx in range(len(bps)-1):
        piece, uci = convert_to_move(bps[moveidx][0], bps[moveidx+1][1], moveidx)
        move = chess.Move.from_uci(uci)
        if ensure_legal:
            legal_moves = list(node.board().generate_legal_moves())
            assert move in legal_moves
        if global_stats:
            color = 'wb'[moveidx%2]
            move_stats[color + piece.lower() + uci[:4]].play_count += 1
            for lmove in legal_moves: # Legal moves
                lboard = node.board()
                lpiece = lboard.piece_at(lmove.from_square).symbol()
                luci = lmove.uci()
                move_stats[color + lpiece.lower() + luci[:4]].legal_count += 1
        node = node.add_variation(move)
    if add_final_move:
        promotion_debug = False
        board = node.board()
        final_move_probs =  get_sorted_final_moves(data)
        final_moves = [mp[0] for mp in final_move_probs]
        # print(final_moves)
        if board.turn == chess.WHITE:
            result = white_result
        else:
            result = -white_result
        for idx, uci in enumerate(final_moves):
            try:
                board.push_uci(uci)
            except ValueError:
                # Assume this is a promotion
                if (uci+'q') not in final_moves:
                    print()
                    print(name, ": ", 'Queen promotion on final move not in training probabilities???')
                    uci = uci + 'q'
                    print("Trying move:", uci)
                    promotion_debug = True
                else:
                    # Must be a knight promotion...
                    print()
                    print(name, ": ", 'Knight promotion on final move')
                    uci = uci + 'n'
                    print("Trying move:", uci)
                    promotion_debug = True
                board.push_uci(uci)
            if board.is_checkmate() and result==1:
                break
            if (board.is_insufficient_material() or
                    board.is_stalemate() or
                    board.can_claim_draw()) and result==0:
                break
            board.pop()
        else:
            print()
            print("Error: Can't find final move")
            print("White result is:", white_result)
            print("Choices for final move:")
            for it in final_move_probs:
                print(it)
            print(game)
            uci = None
            # raise Exception("Can't find final move!")
        if uci is not None:
            move = chess.Move.from_uci(uci)
            node = node.add_variation(move)
        if promotion_debug:
            print(game)
            print("Final move choices were:")
            for it in final_move_probs:
                print(it)
            print("Selected:", uci)
    return str(game)


def write_pgn(pgn, name, pgnfile):
    if pgnfile:
        print(pgn, file=pgnfile)
        print('\n', file=pgnfile)
    else:
        pgnfilename = name + '.pgn'
        pgnfilename = os.path.join(outputdir, pgnfilename)
        with open(pgnfilename, 'w') as pgnfile:
            print(pgn, file=pgnfile)
         
if __name__ == '__main__':
    allnames = set()
    games_in_file = 10000
    pgnfile = None
    try:
        if single_file:
            pgnfilename = os.path.basename(filename).split('.', 1)[0] + '.pgn'
            pgnfilename = os.path.join(outputdir, pgnfilename)
            pgnfile = open(pgnfilename, 'w')
        with tarfile.open(filename) as f:
            for idx, member in enumerate(f):
                if member.name in allnames:
                    raise Exception("Duplicate name in training data???")
                allnames.add(member.name)
                if idx%50==0:
                    print('\n{:5}/{} '.format(idx, games_in_file), end='')
                    if single_file:
                        pgnfile.flush()
                print('.', end='')
                data = f.extractfile(member).read()
                pgn = getpgn(data, member.name)
                if not include_header:
                    # This chops off the header of the pgn string
                    pgn = pgn.rsplit('\n', 1)[-1]
                write_pgn(pgn, member.name, pgnfile)
    finally:
        pgnfile.close()
        
