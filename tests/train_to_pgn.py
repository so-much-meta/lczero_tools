from lcztools.testing import TarTrainingFile
from lcztools.util import tqdm
import fire

def train_to_pgn(filename):
    TarTrainingFile(filename).to_pgn()

if __name__ == '__main__':
    fire.Fire(train_to_pgn)
