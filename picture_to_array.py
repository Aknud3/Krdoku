class PictureToArray:
    """Class for converting a picture to an array."""

    def __init__(self, display, solver_logic, main_menu):
        self.self = self
        self.display = display
        self.solver_logic = solver_logic
        self.main_menu = main_menu
        self.picture_to_array_graphics = r"\textures\picture_to_array_graphics.png"
        self.button_for_uploading_a_picture = None
        self.button_for_returning_to_main_menu = None

    def camera(self):
        """Function for taking a picture."""

        board_state = None
        return board_state

    def run(self):
        """Run function for the app."""
        self.display.run(self.picture_to_array_graphics)
        while True:
            if self.button_for_uploading_a_picture is True:
                board_state = self.camera()
                result = self.solver_logic.run(board_state)
                correct_board_state = result[1]
                self.solver_logic.display_answer(correct_board_state)

            if self.button_for_returning_to_main_menu is True:
                self.main_menu.run()
