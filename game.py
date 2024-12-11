class Game:
    """Main class for the game."""

    def __init__(self, solver_logic, display):
        self.solver_logic = solver_logic
        self.display = display
        self.main_menu = None

        self.game_graphics = "textures\game_graphics.png"
        self.correct_solution_graphics = "textures\correct_solution_graphics.png"
        self.wrong_solution_graphics = "textures\wrong_solution_graphics.png"

        self.button_1 = None
        self.button_2 = None
        self.button_3 = None
        self.button_4 = None
        self.button_5 = None
        self.button_6 = None
        self.button_7 = None
        self.button_8 = None
        self.button_9 = None
        self.button_to_check_solution = None
        self.button_for_eraser = None
        self.button_to_the_main_menu = None
        self.button_for_playing_again = None
        self.button_for_returning_to_main_menu = None

    def check(self, board_state):
        """Check if the solution is correct."""
        result = self.solver_logic.run(board_state)
        is_it_correct = result[0]

        if is_it_correct is True:
            self.display.run(self.wrong_solution_graphics)
        else:
            self.display.run(self.wrong_solution_graphics)

        while True:
            if self.button_for_playing_again is True:
                if is_it_correct is True:
                    board_state = None
                if is_it_correct is True:
                    self.run(board_state)
                else:
                    self.run(board_state)

            if self.button_for_returning_to_main_menu is True:
                self.return_to_main_menu()

    def return_to_main_menu(self):
        """Return to the main menu."""
        self.main_menu.run()

    def place_number(self, number):
        """Place a number on the board."""

    def remove_number(self, x, y):
        """Remove a number from the board."""

    def count_time(self):
        """Count the time."""

    def run(self, board_state):
        """Run function for the game."""
        self.display.run(self.game_graphics)
