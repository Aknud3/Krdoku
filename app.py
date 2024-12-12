# moje funkce
import main_menu
import game
import picture_to_array
import solver_logic


class App:
    """Main class for the application."""

    def __init__(self, main_menu):
        """Initialize the application."""
        self.main_menu = main_menu

    def run(self):
        """Run function for the app."""
        self.main_menu.run()


if __name__ == "__main__":

    main_menu_instance = main_menu.MainMenu(917, 412, None, None)
    game_instance = game.Game(917, 412, None, None)
    picture_to_array_instance = picture_to_array.PictureToArray(917, 412, None, None)

    main_menu_instance.game = game_instance
    game_instance.main_menu = main_menu_instance
    main_menu_instance.game = game_instance

    main_menu_instance.picture_to_array = picture_to_array_instance
    picture_to_array_instance.main_menu = main_menu_instance
    main_menu_instance.picture_to_array = picture_to_array_instance

    app = App(main_menu_instance)
    app.run()
