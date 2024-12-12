import pygame
import button


class PictureToArray:
    """Class for converting a picture to an array."""

    def __init__(self, height, width, solver_logic, main_menu):

        self.height = height
        self.width = width

        self.solver_logic = solver_logic
        self.main_menu = main_menu

        self.background_image = None
        self.screen = None

        self.button_for_returning_to_main_menu = None
        self.button_for_uploading_a_picture = None

    def camera(self):
        """Function for taking a picture."""

    def initialize(self):
        """Initializes the Pygame display."""
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Picture to array")
        self.background_image = pygame.image.load(
            "textures/photo_to_array_graphics.png"
        )
        self.background_image = pygame.transform.scale(
            self.background_image, (self.width, self.height)
        )

    def run(self):
        """Run function for the app."""
        self.button_for_returning_to_main_menu = button.Button(
            "Return to main menu", 239, 738, 140, 140
        )
        self.button_for_uploading_a_picture = button.Button(
            "Upload a picture", 38, 738, 140, 140
        )
        self.initialize()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()

                if self.button_for_returning_to_main_menu.is_clicked(event):
                    self.main_menu.run()
                elif self.button_for_uploading_a_picture.is_clicked(event):
                    print("Upload a picture")

            self.screen.blit(self.background_image, (0, 0))
            self.button_for_returning_to_main_menu.draw(self.screen)
            self.button_for_uploading_a_picture.draw(self.screen)

            pygame.display.flip()
