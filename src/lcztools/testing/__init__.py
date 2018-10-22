from .leela_engine_lc0 import LC0Engine
import importlib
if importlib.find_loader('requests'):
    from .lczero_web import WebMatchGame
    from .lczero_web import WeightsDownloader
from .train_parser import TarTrainingFile
