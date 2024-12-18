# Import pygame, webbrowser libraries and button

import pygame
import button
import webbrowser


class MainMenu:
    """Main class for the main menu."""

    def __init__(
        self,
        height,
        width,
        game,
        picture_to_array,
    ):
        """Initialize the menu."""
        self.height = height
        self.width = width

        self.game = game
        self.picture_to_array = picture_to_array

        self.background_image = None

        self.screen = None

        self.button_for_solving_the_puzzle = None
        self.button_for_playing_the_game = None
        self.button_for_muting_audio = None
        self.button_for_github = None

    def initialize(self):
        """Initializes the Pygame display."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height)) # set dimensions of the window
        pygame.display.set_caption("Main Menu") # Set the caption of window 
        self.background_image = pygame.image.load("textures/main_menu_graphics.png") # load the background image

    def run(self, muted=True): # Muted is on default turn on, so I wont get cancer testing it
        """Run function for the main menu"""

        # Initialize all of the buttons
        self.button_for_playing_the_game = button.Button("game", 56, 649, 300, 150)
        self.button_for_solving_the_puzzle = button.Button("solver", 49, 375, 307, 150)
        self.button_for_muting_audio = button.Button("mute", 316, 830, 80, 80)
        self.button_for_github = button.Button("github", 16, 830, 80, 80)

        self.initialize() # initialize the window

        # Main loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()

                if self.button_for_solving_the_puzzle.is_clicked(event): # Runs Krřešitel
                    running = False
                    self.picture_to_array.run(muted) # Muted is passed down to this

                elif self.button_for_playing_the_game.is_clicked(event): # Runs Klasické Sudoku
                    running = False
                    self.game.run(None, muted) # Muted is passed down to this


                elif self.button_for_muting_audio.is_clicked(event): # Toggle for the mute button
                    if muted:
                        muted = False
                        pygame.mixer.music.unpause()
                        pygame.mixer.music.set_volume(0.5)
                    else:
                        muted = True

                elif self.button_for_github.is_clicked(event): # If the github icon is clicked it opens a github repo in browser
                    webbrowser.open("https://github.com/Aknud3/Krdoku")

            self.screen.blit(self.background_image, (0, 0)) # load the image


            if muted:
                image = pygame.image.load("textures/mute.png") # loads the muted speaker icon 
                self.screen.blit(image, (312, 826))
                pygame.mixer.music.pause()

            # Update the display
            pygame.display.flip()
