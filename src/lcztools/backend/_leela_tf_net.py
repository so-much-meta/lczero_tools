# -*- coding: utf-8 -*-
# Note: yaml config and code to setup tfrprocess borrowed from here:
# https://github.com/glinscott/leela-chess/tree/master/training/tf

import tensorflow as tf
import sys
import os
import sys
import yaml
import textwrap
import gzip
from lcztools.config import get_global_config

# TODO: Hack!!! (This whole file is a hack)
config = get_global_config()
sys.path.append(os.path.expanduser(config.leela_training_tf_dir))

import tfprocess
import tarfile
import numpy as np

from lcztools._weights_file import read_weights_file

YAMLCFG = """
%YAML 1.2
---
name: 'online-64x6'
gpu: 0
dataset:
    num_chunks: 200000
    train_ratio: 0.90
training:
    batch_size: 2048
    total_steps: 60000
    shuffle_size: 1048576
    lr_values:
        - 0.04
        - 0.002
    lr_boundaries:
        - 35000
    policy_loss_weight: 1.0
    value_loss_weight: 1.0
    path: /dev/null
model:
    filters: 64
    residual_blocks: 6
...
"""
YAMLCFG = textwrap.dedent(YAMLCFG).strip()
    


class LeelaModel:
    def __init__(self, weights_filename):
        filters, blocks, weights = read_weights_file(weights_filename)
        cfg = yaml.safe_load(YAMLCFG)
        cfg['model']['filters'] = filters
        cfg['model']['residual_blocks'] = blocks
        cfg['name'] = 'online-{}x{}'.format(filters, blocks)
        print(yaml.dump(cfg, default_flow_style=False))
        
        x = [
            tf.placeholder(tf.float32, [None, 112, 8*8]),
            tf.placeholder(tf.float32, [None, 1858]),
            tf.placeholder(tf.float32, [None, 1])
            ]
        
        self.tfp = tfprocess.TFProcess(cfg)
        self.tfp.init_net(x)
        self.tfp.replace_weights(weights)
    def __call__(self, input_planes):
        input_planes = input_planes.reshape(-1, 112, 8*8)
        policy, value = self.tfp.session.run([self.tfp.y_conv, self.tfp.z_conv],
                                             {self.tfp.x: input_planes, self.tfp.training:False})
        # print("Policy:", policy)
        # print("Value:", value)
        return policy, value


class LeelaLoader:
    @staticmethod
    def from_weights_file(filename, train=False):
        return LeelaModel(filename)
    
