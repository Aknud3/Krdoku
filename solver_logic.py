import random

class SolverLogic:
    """Class for solving the puzzle."""

    def __init__(self):
        self.true = True

    def valid_data(self, list_of_buttons_instance, row, col, num):
        """Check if data is valid for a given cell."""

        for button in list_of_buttons_instance[row]:
            if button.data == num:
                   return False

        for button in [list_of_buttons_instance[r][col] for r in range(9)]:
            if button.data == num:
                    return False

        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                button = list_of_buttons_instance[r][c]
                if button.data == num:
                        return False

        return True

    def create_a_riddle(self, list_of_buttons):
        """Create a Sudoku riddle."""

        for row in list_of_buttons:
            for button_instance in row:
                if button_instance.data is None:
                    numbers = list(range(1, 10))
                    random.shuffle(numbers)

                    for num in numbers:
                            is_valid = self.valid_data(self, list_of_buttons, button_instance.name[1], button_instance.name[0], num)
                            if is_valid:
                                button_instance.data = num
                                button_instance.locked = True
                                if self.create_a_riddle(self, list_of_buttons):
                                     return list_of_buttons
                                
                                button_instance.data = None
                                button_instance.locked = False  
                    return False
                        
        return list_of_buttons