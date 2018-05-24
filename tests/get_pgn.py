from lcztools.testing import TarTrainingFile
from lcztools.util import tqdm
# for game_index, game in tqdm(enumerate(TarTrainingFile('./games14280000.tar.gz'))):
#     game.get_pgn()
TarTrainingFile('./games14280000.tar.gz').to_pgn()