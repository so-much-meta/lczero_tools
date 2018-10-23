import shelve
from lcztools import LeelaBoard
import chess.pgn
import io
import pickle
import sys

# with shelve.open('web_pgn_data.shelf') as s:
#     df_matches = s['df_matches']
#     match_dfs = s['match_dfs']
#     df_pgn = s['df_pgn']

with open('web_pgn.data', 'rb') as f:
    df_matches, match_dfs, df_pgn = pickle.load(f)


for i, (idx, row) in enumerate(df_pgn.iterrows(), 1):
    lcz_board = LeelaBoard()
    pgn_game = chess.pgn.read_game(io.StringIO(row.pgn))
    moves = [move for move in pgn_game.main_line()]
    compressed = []
    features = lcz_board.lcz_features()
    compressed.append(lcz_board.compress_features(features))
    for move in moves[:-1]:
        # print(move)
        lcz_board.push(move)
        features = lcz_board.lcz_features()
        compressed_features = lcz_board.compress_features(features)
        compressed.append(compressed_features)
        # assert(check_compressed_features(lcz_board, compressed_features))
    training_game = (compressed, pgn_game.headers['Result'], row.pgn)
    with open(f'./training_data/match_game_{idx}.data', 'wb') as f:
        f.write(pickle.dumps(training_game))
    print('.', end='')
    if i%100==0:
        print()
    if i%1000==0:
        print(i)
    sys.stdout.flush()

