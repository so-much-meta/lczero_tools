import posixpath
import requests
import bs4
import re
from lcztools import LeelaBoard
from lcztools.util import lazy_property
import chess
import chess.pgn

class WebGame:
    def __init__(self, url):
        '''Create a web match game object.
        
        URL may be the full URL, such as 'http://www.lczero.org/match_game/298660'
        or just a portion, like '298660'. Only the last portion is used'''
        self.url = str(url)

    @lazy_property
    def text(self):
        return requests.get(self.url).text
    
    @lazy_property
    def soup(self):
        return bs4.BeautifulSoup(self.text, 'html.parser')
    
    @lazy_property
    def movelist(self):
        movelist = re.search(r"pgnString: '(.*)'", self.text).group(1) \
            .replace(r'\n', ' ') \
            .replace(r'\x2b', '+') \
            .replace(r'.', '. ') \
            .strip() \
            .split()
        return movelist        
    
    @lazy_property
    def sans(self):
        '''This returns a list of san moves'''
        # Filter out move numbers and result
        sans = [m for m in self.movelist if re.match(r'^[^\d\*]', m)]
        return sans
    
    @lazy_property
    def result(self):
        return self.movelist[-1].replace('\\','')
    
    @lazy_property
    def board(self):
        board = chess.Board()
        for san in self.sans:
            board.push_san(san)
        return board

    @lazy_property
    def leela_board(self):
        board = LeelaBoard()
        for san in self.sans:
            board.push_san(san)
        return board

    @lazy_property
    def pgn_game(self):
        pgn_game = chess.pgn.Game.from_board(self.board)
        if pgn_game.headers['Result'] == '*':
            if self.board.can_claim_draw():
                pgn_game.headers['Result'] = '1/2-1/2'
            elif len(self.board.move_stack) > 400:
                # 450 move rule.. We'll just adjudicate it as a draw if no result and over 400 moves
                pgn_game.headers['Result'] = '1/2-1/2'
        return pgn_game
        
    @lazy_property
    def pgn(self):
        return str(self.pgn_game)
    
    def get_leela_board_at(self, movenum=1, halfmoves=0):
        '''Get Leela board at given move number (*prior* to move)
        
        get_leela_board_at(12, 0): This will get the board on the 12th move, at white's turn
        get_leela_board_at(12, 1): This will get the board on the 12th move, at black's turn
        get_leela_board_at(halfmoves=3): This will return the 4th position (after 3 half-moves)
        '''
        halfmoves = 2*(movenum-1) + halfmoves
        if halfmoves > len(self.sans):
            raise Exception('Not that many moves in game')
        board = LeelaBoard()
        for idx, san in enumerate(self.sans):
            if idx<halfmoves:
                board.push_san(san)
        return board
    
class WebMatchGame(WebGame):
    BASE_URL = 'http://www.lczero.org/match_game'
    def __init__(self, url):
        '''Create a web match game object.
        
        URL may be the full URL, such as 'http://www.lczero.org/match_game/298660'
        or just a portion, like '298660'. Only the last portion is used'''
        url = url.rstrip('/').rsplit('/', 1)[-1]
        super().__init__(posixpath.join(self.BASE_URL, url))
    
class WebTrainingGame(WebGame):
    BASE_URL = 'http://www.lczero.org/game'
    def __init__(self, url):
        '''Create a web match game object.
        
        URL may be the full URL, such as 'http://www.lczero.org/game/298660'
        or just a portion, like '298660'. Only the last portion is used'''
        url = url.rstrip('/').rsplit('/', 1)[-1]
        super().__init__(posixpath.join(self.BASE_URL, url))       