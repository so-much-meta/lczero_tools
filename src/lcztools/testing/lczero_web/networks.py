import bs4
import requests
import posixpath
import os
from collections import OrderedDict
import shutil

from lcztools.config import get_global_config
from lcztools.util import lazy_property


URL_BASE = 'http://lczero.org'

class WeightsDownloader:
    def __init__(self, weights_dir = None, logging = True):
        '''Weights downloader
        
        weights_dir: leave empty to use default config (location for weights files)'''
        if weights_dir is None:
            cfg = get_global_config()
            weights_dir = cfg.weights_dir
        self.weights_dir = weights_dir
        if logging:
            self.log = lambda *args, **kwargs: print(*args, **kwargs)
        else:
            self.log = lambda *args, **kwargs: None
    
    @lazy_property
    def weights_urls(self):
        '''An OrderedDict, latest last'''
        self.log("Getting weights URLs")
        networks_page = requests.get(posixpath.join(URL_BASE, 'networks')).content
        soup = bs4.BeautifulSoup(networks_page, 'html.parser')
        pairs = []
        for it in soup.tbody.find_all('a', download=True):
            url = posixpath.join(URL_BASE, it['href'].lstrip('/'))
            filename = it['download']
            pairs.append((filename, url))
        result = OrderedDict(reversed(pairs))
        self.log("==> Success (Getting weights URLs)")
        return result
    
    @lazy_property
    def latest(self):
        '''Name of latest weights file'''
        return next(reversed(self.weights_urls))
    
    def is_already_downloaded(self, filename):
        fullpath = os.path.join(self.weights_dir, filename)
        return os.path.isfile(os.path.expanduser(fullpath))

    def download_latest_n(self, num, skip_already_downloaded = True):
        '''Download latest n weights file'''
        if num<1:
            raise Exception("Expected a positive number")
        for filename in list(self.weights_urls)[-num:]:
            self.download(filename, skip_already_downloaded)
        
    def download_latest(self, skip_already_downloaded = True):
        '''Download latest weights file'''
        filename = self.latest
        self.download(filename, skip_already_downloaded)
        
    def download_all(self, skip_already_downloaded = True):
        '''Download all weights files
        
        skip_already_downloaded: Skip files that have already been downloaded (default True)'''
        for filename in self.weights_urls:
            self.download(filename, skip_already_downloaded)
        
    def download(self, filename, skip_already_downloaded = True):
        '''Downlaod weights file'''
        self.log(f"Downloading weights file: {filename}")
        if filename not in self.weights_urls:
            raise Exception("Unknown file! {}".format(filename))
        url = self.weights_urls[filename]
        if skip_already_downloaded:
            if self.is_already_downloaded(filename):
                self.log(f"==> Already downloaded (Downloading weights file: {filename})")
                return False
        fullpath = os.path.join(self.weights_dir, filename)
        fullpath_tmp = fullpath + '_download'
        with requests.get(url, stream=True) as r:
            with open(os.path.expanduser(fullpath_tmp), 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        os.rename(fullpath_tmp, fullpath)
        self.log(f"==> Success (Downloading weights file: {filename})")
    