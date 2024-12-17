import pygame

# This is class for the button, that is just transparent thing
# It contains the data and notes 

class Button:
    def __init__(self, name, x, y, width, height):
        self.name = name # Name of the button
        self.rect = pygame.Rect(x, y, width, height) # The positions and width, height

        self.data = None # there is 1,2,3,4,5,6,7,8,9 like the data of number
        self.locked = None # If the button is locked so the data cannot be changed
        self.notes = [] # Here are the notes like [1,3,5]
    
    # This function draw a button on the surface 
    def draw(self, surface):
        # Create a transparent surface 
        temp_surface = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA
        )

        # This set the collor and alpha
        r, g, b = 255, 255, 255  # White color
        alpha = 0  # Adjust transparency here 
        temp_surface.fill((r, g, b, alpha)) # fill the temp_surface with it, so it is transparent

        # Blit the transparent surface onto the main surface 
        surface.blit(temp_surface, (self.rect.x, self.rect.y))

    # This is the function if the button is clicked, that returns True or False
    def is_clicked(self, event):
        if (
            event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
        ):  # Left mouse button
            if self.rect.collidepoint(event.pos):
                return True
        return False