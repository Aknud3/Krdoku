import random
import board

class SolverLogic:
    """Class for solving the puzzle."""

    def __init__(self):
        self.self = self

    def run(self, board_state):
        """Function for solving the puzzle."""
        is_it_correct = None
        board_state = None
        return is_it_correct, board_state

    def display_answer(self, correct_board_state):
        """Function for displaying the answer."""
    
    def create_a_riddle(board):
        board_instance = board
        for button in board_instance.list_of_buttons:
            button.data = 6
            button.locked = True
            board_instance.board_data[button.name[0]][button.name[1]] =  button

        return board