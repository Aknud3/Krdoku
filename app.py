# moje funkce
import main_menu
import game
import display
import picture_to_array
import solver_logic
import button


class App:
    """Main class for the application."""

    def __init__(self, main_menu):
        """Initialize the application."""
        self.main_menu = main_menu

    def run(self):
        """Run function for the app."""
        self.main_menu.run()


if __name__ == "__main__":
    button_instance1 = button.Button(49, 375, 307, 150)
    button_instance2 = button.Button(56, 650, 300, 150)

    display_instance = display.Display()
    solver_logic_instance = solver_logic.SolverLogic()

    game_instance = game.Game(solver_logic_instance, display_instance)

    picture_to_array_instance = picture_to_array.PictureToArray(
        display_instance, solver_logic_instance
    )

    main_menu_instance = main_menu.MainMenu(
        display_instance, button_instance1, button_instance2
    )

    game_instance.main_menu = main_menu_instance
    picture_to_array_instance.main_menu = main_menu_instance

    main_menu_instance.game = game_instance
    main_menu_instance.picture_to_array = picture_to_array_instance

    app = App(main_menu_instance)
    app.run()
