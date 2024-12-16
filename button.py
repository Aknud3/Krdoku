import pygame


class Button:
    def __init__(self, name, x, y, width, height):
        self.name = name
        self.rect = pygame.Rect(x, y, width, height)
        self.data = None
        self.notes = []

    def draw(self, surface):
        # Create a transparent surface
        temp_surface = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA
        )
        # Fill the surface with the button color and alpha value
        r, g, b = 255, 255, 0  # White color
        alpha = 0  # Adjust transparency here (0 fully transparent, 255 fully opaque)
        temp_surface.fill((r, g, b, alpha))
        # Blit the transparent surface onto the main surface
        surface.blit(temp_surface, (self.rect.x, self.rect.y))

    def is_clicked(self, event):
        if (
            event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
        ):  # Left mouse button
            if self.rect.collidepoint(event.pos):
                return True
        return False