'''This will convert a Leela Chess tar training file to PGN.

Usage example:
python train_to_pgn.py games14620000.tar.gz

-- This will output a file called games14620000.pgn
'''


from lcztools.testing import TarTrainingFile
from lcztools.util import tqdm
import fire


def train_to_pgn(train_filename):
    TarTrainingFile(filename).to_pgn()

if __name__ == '__main__':
    fire.Fire(train_to_pgn)
