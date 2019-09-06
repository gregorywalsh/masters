import numpy as np
from connectfour import BoardState
from search import *

def clear_screen():
    print('\n' * 20)

def play_game(board=BoardState(), search_depth=4, players=('cpu', 'cpu'), player=1, search_algo=negamax, ui=True):

    def get_winner():
        if board.end_state() == 'lose':
            winner = 1 if board.last_player == 1 else 2
        else:
            winner = None
        return winner

    def print_winner():

        winner = get_winner()
        if winner:
            print('=' * 8, 'Player', winner, 'Wins!', '=' * 8, )
        else:
            print('=' * 10, 'It''s a draw', '=' * 10)

    def get_action():

        def get_input():
            action = None

            player_num = '1' if player == 1 else '2'

            while action is None:

                try:
                    user_input = int(input('P%s - Which column?\n' % player_num))
                    if 1 <= user_input <= BoardState.WIDTH:
                        action = user_input - 1
                    else:
                        print('Oops - that''s not a valid number. Enter 1-%d' % BoardState.WIDTH)
                except ValueError:
                    print('Oops - that''s not a valid number. Enter 1-%d' % BoardState.WIDTH)
            return action

        if player == 1:
            if players[0] == 'human':
                action = get_input()
            else:
                action = search_algo(state=board, player=player, remaining_plys=search_depth)[1]
        else:
            if players[1] == 'human':
                action = get_input()
            else:
                action = search_algo(state=board, player=player, remaining_plys=search_depth)[1]

        return action

    if ui:
        clear_screen()
        print(board, '\n')

    while board.end_state() is None:

        action = get_action()

        if ui: clear_screen()
        board = board.execute_action(action)

        if ui: print(board, '\n')

        player = -player

    if ui: print_winner()

    return get_winner(), search_algo.counter

search_depth = 3
players = ('cpu', 'cpu')
search_algos = [negamax, negamax_ab, negamax_ab_with_ordering]

for search_depth in range(1, 8):
    game_results = []
    iterations = []
    print('Depth:', search_depth)
    for algo in search_algos:
        for i in range(50):
            algo.counter = 0
            winner, counter = play_game(search_depth=search_depth, players=players, search_algo=algo, ui=False)
            game_results.append(winner)
            iterations.append(algo.counter)

        print('Algo:', algo.__name__)
        print('Negamax iter:', iterations)
        print('Negamax results:', game_results)
        print()


"""
.  X  .  X  .  X  X
.  X  .  X  .  X  O
.  O  .  O  .  O  X
.  X  .  X  .  X  O
O  O  .  X  O  O  O
O  O  X  O  X  O  O
"""

board = [
    [1, 1, 0, 0, 0, 0],
    [1, 1, -1, 1, -1, -1],
    [-1, 0, 0, 0, 0, 0],
    [1, -1, -1, 1, -1, -1],
    [-1, 1, 0, 0, 0, 0],
    [1, 1, -1, 1, -1, -1],
    [1, 1, 1, -1, 1, -1]
]

board = np.array(board, dtype=np.int8)
col_counts = [2, 6, 1, 6, 2, 6, 6]
player = -1
board = BoardState(board=board, column_counts=col_counts,player=-player, last_action_coords=None)
search_algo = negamax

score, action = search_algo(state=board, player=player, remaining_plys=search_depth)
print('Algorithm:', search_algo.__name__)
print('Alpha: -inf')
print('Beta: inf')
print('Num iterations:', search_algo.counter)
print('Player 1 is ''O'', Player 2 is ''X''', '\n')
print(board)
print_graph(search_algo.edges, score, action, player)