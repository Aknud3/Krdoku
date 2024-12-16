import pygame
import button
import webbrowser


class MainMenu:
    """Main class for the application."""

    def __init__(
        self,
        height,
        width,
        game,
        picture_to_array,
    ):
        """Initialize the application."""
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
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Main Menu")
        self.background_image = pygame.image.load("textures/main_menu_graphics.png")
        self.background_image = pygame.transform.scale(
            self.background_image, (self.width, self.height)
        )

    def run(self, muted=True):
        """Run function for the app."""
        self.button_for_playing_the_game = button.Button("game", 56, 649, 300, 150)
        self.button_for_solving_the_puzzle = button.Button("solver", 49, 375, 307, 150)
        self.button_for_muting_audio = button.Button("mute", 316, 830, 80, 80)
        self.button_for_github = button.Button("github", 16, 830, 80, 80)

        self.initialize()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()

                if self.button_for_solving_the_puzzle.is_clicked(event):
                    running = False
                    self.picture_to_array.run(muted)

                elif self.button_for_playing_the_game.is_clicked(event):
                    running = False
                    self.game.run(None, muted)

                elif self.button_for_muting_audio.is_clicked(event):
                    if muted:
                        muted = False
                        pygame.mixer.music.unpause()
                    else:
                        muted = True

                elif self.button_for_github.is_clicked(event):
                    webbrowser.open("https://github.com/Aknud3/Krdoku")

            self.screen.blit(self.background_image, (0, 0))

            # Draw all buttons
            self.button_for_playing_the_game.draw(self.screen)
            self.button_for_solving_the_puzzle.draw(self.screen)

            if muted:
                image = pygame.image.load("textures/mute.png")
                self.screen.blit(image, (312, 826))
                pygame.mixer.music.pause()

            # Update the display
            pygame.display.flip()
