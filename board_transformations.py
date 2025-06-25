"""
Functions for transforming the Tic Tac Toe board state into all possible transformations.
This includes rotations and reflections to generate equivalent board states.
"""

import numpy as np
from tictactoe import TicTacToeEnv

def get_all_transformations(board):
    """
    Given a 3x3 board as a 2D NumPy array, return a list of all 8 symmetries:
    4 rotations and their reflections.
    """
    transforms = []
    # Rotations
    for k in range(4):
        rot = np.rot90(board, k)
        transforms.append(rot)
        # Reflection of rotation
        transforms.append(np.fliplr(rot))
    return transforms

def canonical_board(board_tuple):
    """
    Convert a flat board tuple of length 9 into its canonical form
    under symmetries (the lexicographically smallest tuple).
    """
    board = np.array(board_tuple).reshape((3,3))
    all_boards = get_all_transformations(board)
    # Flatten each and convert to tuple
    flattened = [tuple(b.flatten()) for b in all_boards]
    # Return the minimal tuple
    return min(flattened)