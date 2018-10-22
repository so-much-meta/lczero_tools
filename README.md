# lczero-tools
Python utilities for experimenting with Leela Chess Zero a neural network based chess engine: https://github.com/glinscott/leela-chess/

#### Note: This is primarily for looking at the Leela Chess neural network itself, outside of search/MCTS (although search may be added eventually).

This makes heavy use of python-chess located at https://github.com/niklasf/python-chess

The current implementation is primarily geared towards pytorch, but tensorflow is possible using the training/tf portion of leela-chess.

Example usage (also see /tests/*.py and [Examples.ipynb](https://github.com/so-much-meta/lczero_tools/blob/master/notebooks/Examples.ipynb)):
```python
>>> from lcztools import load_network, LeelaBoard
>>> # Note: use pytorch_cuda for cuda support
>>> net = load_network('pytorch_cuda', 'weights.txt.gz')
>>> board = LeelaBoard()
>>> # Many of Python-chess's methods are passed through, along with board representation
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

## INSTALL
```
# With both torch and util dependencies for NN evaluation
pip install git+https://github.com/so-much-meta/lczero_tools.git#egg=lczero-tools[torch,util]
# Or just util extras (parse training games, run lczero engine, etc)
pip install git+https://github.com/so-much-meta/lczero_tools.git#egg=lczero-tools[util]


git clone https://github.com/so-much-meta/lczero_tools
cd lczero_tools
# Note: Creating and using a virtualenv or Conda environment before install is suggested, as always
pip install .
# Or for developer/editable install, to make in place changes:
# pip install -e .
```

## TODO
1. [x] **DONE:** Implement testing to verify position evaluations match lczero engine.
   * [ ] Using /tests/test_net_eq_engine.py, results look good. But specific PGNs might be helpful too.
2. [x] **DONE:** Add config mechanism and Jupyter notebook examples
3. [x] **DONE:** Add training data parser module. Use cases are:
   * [x] **DONE:** Training data to PGN
   * [ ] Verification of training data correctness.
   * [ ] Loss calculation - allow comparison between networks on same data
4. [x] **DONE:** lczero web scraping *(NOT FOR HEAVY USE)*
   * [x] **DONE:** Convert individidual match and training games to PGN (URL => PGN)
   * [x] **DONE:** Download weights files
5. [ ] OpenCL support! This should be possible with https://github.com/plaidml/plaidml
6. [ ] Investigate optimizations (CUDA, multiprocessing, etc). Goal is to eventually have a fast enough python-based implementation to do MCTS and get decent nodes/second comparable to Leela's engine -- in cases where neural network eval speed is the bottleneck.
   * [ ] However, no optimizations should get (too much) in the way of clarity or ease of changing code to do experiments.
7. [ ] Possible MCTS implementation
