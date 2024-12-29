# Import all of the functions
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

        pygame.mixer.init() # define pygame mixer
        pygame.mixer.music.load("background_music.mp3") # load background music
        pygame.mixer.music.set_volume(0.0) # set the volume to 0, because default is turn off and it is controlled in main menu
        pygame.mixer.music.play(loops=-1, start=0.0) # play it in a loop

    def run(self):
        """Run the main_menu.run so it will just display the main menu."""
        self.main_menu.run()

    def stop_music(self):
        """Stop the background music."""
        pygame.mixer.music.stop()

if __name__ == "__main__": # If the name is main so there is not some hacking stuff
    board_instance = board.Board() # create a board for the game
    main_menu_instance = main_menu.MainMenu(917, 412, None, None) # create a main_menu with 2 blank variables
    game_instance = game.Game(917, 412, None, board_instance) # create a game with 2 blank variables
    picture_to_array_instance = picture_to_array.PictureToArray(917, 412, None, None) # create the last function with 2 blank variables

    # I will just add the blank variables to the things I inicialized so it will all be linked
    main_menu_instance.game = game_instance 
    game_instance.main_menu = main_menu_instance
    main_menu_instance.game = game_instance
    main_menu_instance.picture_to_array = picture_to_array_instance
    picture_to_array_instance.main_menu = main_menu_instance
    main_menu_instance.picture_to_array = picture_to_array_instance

    app = App(main_menu_instance) # inicialize App 
    try:
        app.run() # run the app if you can
    finally:
        app.stop_music() # stop the music

    pygame.quit()  # quit pygame
