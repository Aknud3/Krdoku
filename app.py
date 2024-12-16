# moje funkce
import pygame
import main_menu
import game
import picture_to_array
import board


class App:
    """Main class for the application."""

    def __init__(self, main_menu_for_app):
        """Initialize the application."""
        self.main_menu = main_menu_for_app

        # Initialize Pygame mixer
        pygame.mixer.init()

        # Load the background music
        pygame.mixer.music.load("background_music.mp3")

        pygame.mixer.music.set_volume(0.5)
        # Play the background music in a loop (-1 means infinite loop)
        pygame.mixer.music.play(loops=-1, start=0.0)

    def run(self):
        """Run function for the app."""
        self.main_menu.run()

    def stop_music(self):
        """Stop the background music when the app exits."""
        pygame.mixer.music.stop()


if __name__ == "__main__":
    board_instance = board.Board(None)
    main_menu_instance = main_menu.MainMenu(917, 412, None, None)
    game_instance = game.Game(917, 412, None, None, board_instance)
    picture_to_array_instance = picture_to_array.PictureToArray(917, 412, None, None)

    main_menu_instance.game = game_instance
    game_instance.main_menu = main_menu_instance
    main_menu_instance.game = game_instance

    main_menu_instance.picture_to_array = picture_to_array_instance
    picture_to_array_instance.main_menu = main_menu_instance
    main_menu_instance.picture_to_array = picture_to_array_instance

    app = App(main_menu_instance)
    try:
        app.run()
    finally:
        # Ensure the music stops when the app exits
        app.stop_music()

    pygame.quit()  # Quit Pygam
