"""This will convert a polyglot book to a list of move sequences and corresponding probabilities
given that moves are selected in proportion to their polyglot weight (i.e., win-rate)

E.g.:
"""


import chess
import chess.polyglot
from collections import namedtuple
import fire
import math
import numpy as np

Entry = namedtuple('Entry', 'move weight')

def dfs(reader, board, depth, ll=0):
    """Yield a list of move-sequences and corresponding log-likelihoods"""
    total_weight = 0
    entries = []
    for pentry in reader.find_all(board):
        total_weight += pentry.weight
        entries.append(Entry(pentry.move(), pentry.weight))
    for entry in entries:
        board.push(entry.move)
        ll_cur = ll + math.log(entry.weight/total_weight)
        if depth<=1:
            moves = ' '.join(move.uci() for move in board.move_stack)
            yield((moves, ll_cur))
        else:
            yield from dfs(reader, board, depth-1, ll_cur)
        board.pop()

def main(book_filename, length):
    """Output a list of fixed-length move-sequences and corresponding probabilities
    in a polyglot opening book
    
    book_filename: Filename of polyglot opening book
    length: Length of move sequences to extract
    """
    assert(length>=1)
    with chess.polyglot.open_reader("bookfish.bin") as reader:
        moves, lls = zip(*dfs(reader, chess.Board(), length))
    lls = np.array(lls)  # log likelihoods
    ls = np.exp(lls - max(lls))  # likelihoods
    ps = ls/sum(ls)  # probabilities
    seqs = zip(ps, moves)
    seqs = sorted(seqs, key=lambda it: it[0], reverse=True)
    for item in seqs:
        print("{:.9e} {}".format(*item))
        
# main("bookfish.bin", 8)
fire.Fire(main)