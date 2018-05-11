# lczero_tools
Utilities for experimenting with leela-chess-zero, a neural network based chess engine: https://github.com/glinscott/leela-chess/

This makes heavy use of python-chess located at https://github.com/niklasf/python-chess

The network may be run with pytorch, or tensorflow (tensorflow implementation currently imports from leela-chess training code)

For now, the following is possible (also see test.py):
```
>>> from lcztools import load_network, LeelaBoard
>>> net = load_network('pytorch', 'weights.txt.gz')
>>> board = LeelaBoard()
>>> board.push_uci('e2e4')
>>> print(board)
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R
Turn: Black
>>> policy, value = net.evaluate(board)
>>> print(policy)
OrderedDict([('c7c5', 0.5102739), ('e7e5', 0.16549255), ('e7e6', 0.11846365), ('c7c6', 0.034872748),
('d7d6', 0.025344277), ('a7a6', 0.02313047), ('g8f6', 0.021814445), ('g7g6', 0.01614216), ('b8c6', 0.013772337),
('h7h6', 0.013361011), ('b7b6', 0.01300134), ('d7d5', 0.010980369), ('a7a5', 0.008497312), ('b8a6', 0.0048270077),
('g8h6', 0.004309486), ('f7f6', 0.0040882644), ('h7h5', 0.003910391), ('b7b5', 0.0027878743), ('f7f5', 0.0025032777),
('g7g5', 0.0024271626)])
>>> print(value)
0.4715215042233467
```
