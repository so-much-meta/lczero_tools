from .leela_engine import LCZeroEngine
import importlib
if importlib.find_loader('requests'):
    from .lczero_web import WebMatchGame
    from .lczero_web import WeightsDownloader
from .training_data import TarTrainingFile
