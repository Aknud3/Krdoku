class MainMenu:
    """Main class for the application."""

    def __init__(
        self, display, button_for_solving_the_puzzle, button_for_playing_the_game
    ):
        """Initialize the application."""
        self.display = display
        self.main_menu_graphics = (
            "textures/main_menu_graphics.png"  # Fixed path separator
        )
        self.button_for_solving_the_puzzle = button_for_solving_the_puzzle
        self.button_for_playing_the_game = button_for_playing_the_game

    def run(self):
        """Run function for the app."""
        # Load the background image (returns a Surface)
        screen = self.display.run(self.main_menu_graphics)

        # Main loop for the menu
        running = True
        while running:
            # Handle events and draw the buttons
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Draw the background image
            self.display.screen.blit(screen, (0, 0))

            # Draw buttons on top of the image
            self.button_for_playing_the_game.draw(self.display.screen)
            self.button_for_solving_the_puzzle.draw(self.display.screen)

            # Check if buttons are clicked
            if self.button_for_playing_the_game.is_pressed():
                # Start the game when the play button is pressed
                self.game.run()

            if self.button_for_solving_the_puzzle.is_pressed():
                # Start the puzzle solving logic when the solve button is pressed
                self.picture_to_array.run()

            # Update the display
            pygame.display.flip()
