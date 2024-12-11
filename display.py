import sys
import pygame


class Display:
    """Class for displaying the graphics of everything."""

    def __init__(self, width=412, height=917):
        self.width = width
        self.height = height
        self.screen = None
        self.image_surface = None  # To hold the image surface for later use

    def initialize(self):
        """Initializes the Pygame display."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Display Graphics")

    def run(self, graphics):
        """Function for displaying the graphics and returning the surface.
        Args:
            graphics (str): Path to the image file to be displayed.
        """
        # Initialize display if not already initialized
        if self.screen is None:
            self.initialize()

        # Load the image
        try:
            self.image_surface = pygame.image.load(graphics)
        except pygame.error as e:
            print(f"Failed to load image: {e}")
            return None

        # Main loop for displaying the image
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Draw the image at the center of the screen
            x = (self.width - self.image_surface.get_width()) // 2
            y = (self.height - self.image_surface.get_height()) // 2
            self.screen.blit(self.image_surface, (x, y))

            # Update the display
            pygame.display.flip()

        # Return the image surface for later usage
        return self.image_surface
