class MainMenu:
    """Main class for the application.
    """
    def __init__(self, game, picture_to_array, display):
        """ Initialize the application.
        """
        self.self = self
        self.game = game
        self.picture_to_array = picture_to_array
        self.display = display
        self.main_menu_graphics = r"\textures\main_menu_graphics.png"
        self.button_for_solving_the_puzzle = None
        self.button_for_playing_the_game = None
        
    def run(self):
        """ Run function for the app.
        """
        self.display.run(self.main_menu_graphics)
        while True:
            if self.button_for_playing_the_game is True:
                self.game.run()
            if self.button_for_solving_the_puzzle is True:
                self.picture_to_array.run()