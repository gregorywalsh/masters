import numpy as np
from operator import itemgetter

class BoardState():

    WIDTH = 7
    HEIGHT = 6
    DIRECTIONS = ((1,0), (0,1), (1, 1), (1, -1))
    WIN_MOVE_COUNTS = np.array(
        [
            [3, 4, 5, 5, 4, 3],
            [4, 6, 8, 8, 6, 4],
            [5, 8, 11, 11, 8, 5],
            [7, 10, 13, 13, 10, 7],
            [5, 8, 11, 11, 8, 5],
            [4, 6, 8, 8, 6, 4],
            [3, 4, 5, 5, 4, 3]
        ],
        dtype=np.int8
    )

    counter = -1

    def __init__(self, board=None, column_counts=None, player=None, last_action_coords=None):
        """
        :param board: A byte array
        """
        if board is not None:
            self.board = board
            self.column_counts = column_counts
            self.last_player = player
            self.next_player = -self.last_player
            self.last_action_coords = last_action_coords
        else:
            self.board = np.zeros(shape=(BoardState.WIDTH, BoardState.HEIGHT), dtype=np.int8)
            self.column_counts = tuple(0 for i in range(BoardState.WIDTH))
            self.last_player = -1
            self.next_player = -self.last_player
            self.last_action_coords = None

        self.id = BoardState.counter
        BoardState.counter += 1

    def evaluate(self):
        """
        Evaluates the current position
        :return: A signed integer. Higher is better for Player-1
        """

        return np.sum(np.multiply(BoardState.WIN_MOVE_COUNTS, self.board))

    def possible_actions(self):
        """
        Gets a list of all legal moves
        :return: A list of actions
        """

        actions = (i for i, count in enumerate(self.column_counts) if count < BoardState.HEIGHT)
        return actions

    def sorted_possible_actions(self):
        """
        Gets a list of all legal moves
        :return: A list of actions
        """

        actions = [i for i, count in enumerate(self.column_counts) if count < BoardState.HEIGHT]
        actions.sort(key=lambda action: BoardState.WIN_MOVE_COUNTS[action, self.column_counts[action]], reverse=True)
        return actions

    def execute_action(self, action):
        """
        Executes an action and returns a new board state
        :return: A new BoardState object
        """
        assert self.column_counts[action] < BoardState.HEIGHT, "Cannot make a move in this column"

        new_col_counts = tuple(count + 1 if i == action else count for i, count in enumerate(self.column_counts))

        new_board = np.copy(self.board)
        new_board[action, self.column_counts[action]] = self.next_player

        return BoardState(new_board, new_col_counts, self.next_player, (action, new_col_counts[action] - 1))

    def end_state(self):
        """
        Determines if the game has come to an end
        :return:
        """

        if self.last_action_coords:
            # CHECK WIN
            for direction in BoardState.DIRECTIONS:
                num_in_a_row = 0
                for i in range(-3, 4):
                    offset = (direction[0] * i, direction[1] * i)
                    nearby_chip_coords = (offset[0] + self.last_action_coords[0], offset[1] + self.last_action_coords[1])
                    if 0 <= nearby_chip_coords[0] < BoardState.WIDTH and 0 <= nearby_chip_coords[1] < BoardState.HEIGHT:
                        if self.board[nearby_chip_coords[0], nearby_chip_coords[1]] == self.last_player:
                            num_in_a_row += 1
                        else:
                            num_in_a_row = 0

                        if num_in_a_row == 4:
                            return 'lose'

            # CHECK DRAW AFTER CHECKING WIN
            if all(count == BoardState.HEIGHT for count in self.column_counts):
                return 'draw'

        # OTHERWISE
        return None

    def __str__(self):
        chars = ('.', 'O', 'X')
        board_str = ''
        for row in reversed(tuple(zip(*self.board))):
            board_str += '  '.join((chars[slot] for slot in row)) + '\n'
        return board_str
