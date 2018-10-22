import requests
from bs4 import BeautifulSoup
import pandas as pd
from lcztools.testing import WebMatchGame
from collections import OrderedDict
import shelve
import sys

with shelve.open('web_pgn_data.shelf') as s:
    if 'df_matches' in s:
        df_matches = s['df_matches'] 
        match_dfs = s['match_dfs']
        df_pgn = s['df_pgn']

if 'df_pgn' in dir():
    print("Currently {} games in shelf".format(len(df_pgn)))
else:
    print("No games yet in shelf...")
# Everytime I run this, I'll grab an extra GAMES_TO_GRAB games
GAMES_TO_GRAB = 10000


def get_table(url):
    global soup
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    head = soup.thead.tr
    head_cols = [c.text for c in head.find_all('th')]
    head_cols.append('hrefs')

    rows = []
    for tr in soup.tbody.find_all('tr'):
        cells = [c.text for c in tr.find_all('td')]
        cells.append([c['href'] for c in tr.find_all('a')])
        rows.append(cells)
    df = pd.DataFrame(rows, columns=head_cols)
    return df

df_matches = get_table('http://www.lczero.org/matches/1')
df_matches = df_matches.astype({'Id':int, 'Run':int, 'Candidate':int, 'Current':int}).set_index('Id')
df_matches = df_matches.sort_index(ascending=False)           


if 'match_dfs' not in dir():
    match_dfs = OrderedDict()

if 'df_pgn' not in dir():
    print("Creating df_pgn")
    df_pgn = pd.DataFrame(index=pd.Index([], name='match_game_id', dtype=int), columns=['match_id', 'pgn'])
    df_pgn = df_pgn.astype({'match_id':int})

all_games_received = set(df_pgn.index)
num_pgns_grabbed = 0
for match_id, row in df_matches.iterrows():
    href = row.hrefs[0]
    if match_id in match_dfs:
        print("Already retrieved", href)
    else:
        print(href)
        match_dfs[match_id] = get_table(f'http://www.lczero.org/{href.lstrip("/")}')
        match_dfs[match_id] = match_dfs[match_id].astype({'Game Id':int}).set_index('Game Id')
    match_dfs[match_id] = match_dfs[match_id].sort_index(ascending=False)
    df = match_dfs[match_id]
    for game_id, row in df.iterrows():
        if game_id in all_games_received:
            continue
        href = row.hrefs[0]
        print(game_id, end=', ')
        sys.stdout.flush()
        wmg = WebMatchGame(href)
        pgn = wmg.pgn
        df_pgn.loc[game_id] = match_id, pgn
        all_games_received.add(game_id)
        num_pgns_grabbed += 1
        if num_pgns_grabbed >= GAMES_TO_GRAB:
            break
    if num_pgns_grabbed >= GAMES_TO_GRAB:
        break

print("Done... Downloaded {} files. Saving back to shelf".format(num_pgns_grabbed))

with shelve.open('web_pgn_data.shelf') as s:
    s['df_matches'] = df_matches
    s['match_dfs'] = match_dfs
    s['df_pgn'] = df_pgn

print("Currently {} games in shelf".format(len(df_pgn)))