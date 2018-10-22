'''Config handling

Config is entered into an 'lcztools.ini' file, which is searched for in:
1. Script path (via sys.argv[0])
2. Current-working-directory
3. Home directory
'''

import configparser
import sys
import os

class LCZToolsConfig:
    def __init__(self, dct={}):        
        # This is directory where the network weights are stored
        self.weights_dir = dct.get('weights_dir') or '.'
        
        # This is the default weights filename to use, if none provided
        self.weights_file = dct.get('weights_file') or 'weights.txt.gz'
        
        # This is the default backend to use, if none provided
        self.backend = dct.get('backend') or 'pytorch'
        
        ## No longer supported
        ## This is the lczero engine to use, only currently used for testing/validation
        # self.lczero_engine = dct.get('lczero_engine')

        # This is the lc0 engine to use, only currently used for testing/validation
        self.lc0_engine = dct.get('lc0_engine')

        # This is, e.g. ~/sompath/leela-chess/training/tf -- currently only used as hackish tensorflow
        # mechanism, which is not used with pytorch backend
        self.leela_training_tf_dir = dct.get('leela_training_tf_dir')

        # Policy softmax temp
        self.policy_softmax_temp = dct.get('policy_softmax_temp') or 1.0
        self.policy_softmax_temp = float(self.policy_softmax_temp)
    
    def get_weights_filename(self, basename=None):
        '''Get weights filename given a base filename, or use default self.weights_file
        If filename provided is absolute, just return that.
        This way, lcztools.load_network('myweights.txt.gz') returns relative to weights_dir,
        but user may still use a full path name. 
        '''
        if basename:
            basename = os.path.expanduser(basename)
            if os.path.isabs(basename):
                return basename
        basename = basename or self.weights_file
        return os.path.join(os.path.expanduser(self.weights_dir), basename)
                

def find_config_file():
    '''Search for an lcztools.ini file, and return full pathname'''
    # 1. Search in main script directory
    filename = 'lcztools.ini'
    dirname = os.path.dirname(sys.argv[0])
    dirname = os.path.abspath(dirname)
    fname = os.path.join(dirname, filename)
    if os.path.isfile(fname):
        return fname
    # 2. Search in cwd
    dirname = os.path.abspath('.')
    fname = os.path.join(dirname, filename)
    if os.path.isfile(fname):
        return fname
    # 3. Search in homedir
    dirname = os.path.expanduser('~')
    fname = os.path.join(dirname, filename)
    if os.path.isfile(fname):
        return fname
    # No config file found..
    return None


def set_global_config(filename=None, section='default'):
    '''Set global config. If filename is not provided, then attempt to find an lcztools.ini file'''
    global _global_config
    if filename is None:
        filename = find_config_file()
    if filename is None:
        # Empty config
        _global_config = LCZToolsConfig()
        return
    config = configparser.ConfigParser()
    config.read(filename)
    _global_config = LCZToolsConfig(config[section])


def get_global_config():
    global _global_config
    if _global_config is None:
        set_global_config()
    return _global_config
    
_global_config = None
