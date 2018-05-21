# lczero_tools
Python utilities for experimenting with leela-chess-zero, a neural network based chess engine: https://github.com/glinscott/leela-chess/

This makes heavy use of python-chess located at https://github.com/niklasf/python-chess

The network may be run with pytorch, or tensorflow (tensorflow implementation currently imports from leela-chess training code)

For now, the following is possible (also see /tests/*.py):
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

## INSTALL
```
git clone https://github.com/so-much-meta/lczero_tools
cd lczero_tools
pip install .
# Or for developer/editable install, to make in place changes:
# pip install -e .
```

## TODO
1. ~~Implement testing to verify position evaluations match lczero engine.~~
   * Using /tests/test_net_eq_engine.py, results look good. But specific PGNs might be helpful too.
2. Add training data parser module. Use cases are:
   * Training data to PGN
   * Verification of training data correctness.
   * Loss calculation - allow comparison between networks on same data
3. OpenCL support! This should be possible with https://github.com/plaidml/plaidml
4. Investigate optimizations (CUDA, multiprocessing, etc). Goal is to eventually have a fast enough python-based implementation to do MCTS and get decent nodes/second comparable to Leela's engine -- in cases where neural network eval speed is the bottleneck.
   * However, no optimizations should get (too much) in the way of clarity or ease of changing code to do experiments.

Note: In order to make this work with tensorflow CPU-only mode using leela-chess tfprocess, changes had to be made for dimension ordering of the input (most likely this change slows things down a lot)...
```
diff --git a/training/tf/tfprocess.py b/training/tf/tfprocess.py
index 97f04a2..be79868 100644
--- a/training/tf/tfprocess.py
+++ b/training/tf/tfprocess.py
@@ -49,8 +49,11 @@ def bn_bias_variable(shape):
     return tf.Variable(initial, trainable=False)
 
 def conv2d(x, W):
-    return tf.nn.conv2d(x, W, data_format='NCHW',
+    x = tf.transpose(x, [0, 2, 3, 1])
+    x = tf.nn.conv2d(x, W, data_format='NHWC',
                         strides=[1, 1, 1, 1], padding='SAME')
+    x = tf.transpose(x, [0, 3, 1, 2])
+    return x
 
 class TFProcess:
     def __init__(self, cfg):
@@ -270,7 +273,7 @@ class TFProcess:
             save_path = self.saver.save(self.session, path, global_step=steps)
             print("Model saved in file: {}".format(save_path))
             leela_path = path + "-" + str(steps) + ".txt"
-            self.save_leelaz_weights(leela_path) 
+            self.save_leelaz_weights(leela_path)
             print("Weights saved in file: {}".format(leela_path))
 
     def save_leelaz_weights(self, filename):
@@ -328,7 +331,7 @@ class TFProcess:
             h_bn = \
                 tf.layers.batch_normalization(
                     conv2d(inputs, W_conv),
-                    epsilon=1e-5, axis=1, fused=True,
+                    epsilon=1e-5, axis=1, fused=False,
                     center=False, scale=False,
                     training=self.training)
         h_conv = tf.nn.relu(h_bn)
@@ -358,7 +361,7 @@ class TFProcess:
             h_bn1 = \
                 tf.layers.batch_normalization(
                     conv2d(inputs, W_conv_1),
-                    epsilon=1e-5, axis=1, fused=True,
+                    epsilon=1e-5, axis=1, fused=False,
                     center=False, scale=False,
                     training=self.training)
         h_out_1 = tf.nn.relu(h_bn1)
@@ -366,7 +369,7 @@ class TFProcess:
             h_bn2 = \
                 tf.layers.batch_normalization(
                     conv2d(h_out_1, W_conv_2),
-                    epsilon=1e-5, axis=1, fused=True,
+                    epsilon=1e-5, axis=1, fused=False,
                     center=False, scale=False,
                     training=self.training)
         h_out_2 = tf.nn.relu(tf.add(h_bn2, orig))
```
