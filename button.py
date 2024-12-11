import pygame


class Button:
    def __init__(self, x, y, width, height, action=None):
        """
        Initializes the button with an optional action.

        Args:
            x (int): X-coordinate of the top-left corner of the button.
            y (int): Y-coordinate of the top-left corner of the button.
            width (int): Width of the button.
            height (int): Height of the button.
            action (callable, optional): Function to call when the button is clicked.
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.action = action

    def draw(self, screen):
        """
        Draws the transparent button on the screen.

        Args:
            screen (pygame.Surface): The surface on which to draw the button.
        """
        transparent_surface = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA
        )
        transparent_surface.fill((0, 0, 0, 0))  # Fully transparent surface
        screen.blit(transparent_surface, (self.rect.x, self.rect.y))

    def handle_event(self, event):
        """
        Handles the event when the button is clicked.

        Args:
            event (pygame.event.Event): The event object to check for clicks.

        Returns:
            bool: True if the button was clicked, False otherwise.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                if self.action:  # If there's an action, execute it
                    self.action()
                return True
        return False
