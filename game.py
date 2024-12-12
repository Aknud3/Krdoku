import pygame
import button


class Game:
    """Main class for the game."""

    def __init__(self, height, width, solver_logic, main_menu):

        self.height = height
        self.width = width

        self.solver_logic = solver_logic
        self.main_menu = main_menu

        self.background_image = None
        self.screen = None

        self.board_surface = None

        self.button_for_checking = None
        self.button_for_eraser = None
        self.button_for_main_menu = None
        self.button_for_number_1 = None
        self.button_for_number_2 = None
        self.button_for_number_3 = None
        self.button_for_number_4 = None
        self.button_for_number_5 = None
        self.button_for_number_6 = None
        self.button_for_number_7 = None
        self.button_for_number_8 = None
        self.button_for_number_9 = None

    def create_a_riddle(self):
        """Create a riddle."""

    def check(self, board_state):
        """Check if the solution is correct."""

    def place_number(self, number):
        """Place a number on the board."""

    def remove_number(self, x, y):
        """Remove a number from the board."""

    def draw_board_clear(self, screen, surface):
        """Draw the board."""
        # Fill background
        surface.fill((255, 255, 255))

        # Draw the grid lines
        for i in range(9 + 1):  # Include the outer boundary
            # Decide line thickness
            if i % 3 == 0:
                thickness = 4  # Thicker line for 3x3 grid blocks
            else:
                thickness = 1  # Regular line for smaller cells

            # Draw horizontal lines
            pygame.draw.line(surface, (0, 0, 0), (0, i * 45), (400, i * 45), thickness)
            # Draw vertical lines
            pygame.draw.line(surface, (0, 0, 0), (i * 45, 0), (i * 45, 400), thickness)

        screen.blit(surface, (6, 228))

    def initialize(self):
        """Initializes the Pygame display."""
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Game")
        self.background_image = pygame.image.load("textures/game_graphics.png")
        self.background_image = pygame.transform.scale(
            self.background_image, (self.width, self.height)
        )

    def run(self, board_state):
        """Run function for the game."""
        self.button_for_checking = button.Button("check", 292, 643, 76, 76)
        self.button_for_eraser = button.Button("eraser", 292, 740, 76, 76)
        self.button_for_main_menu = button.Button("main_menu", 292, 837, 76, 76)

        self.button_for_number_1 = button.Button("1", 18, 653, 81, 80)
        self.button_for_number_2 = button.Button("2", 102, 653, 81, 80)
        self.button_for_number_3 = button.Button("3", 187, 653, 81, 80)
        self.button_for_number_4 = button.Button("4", 18, 738, 81, 80)
        self.button_for_number_5 = button.Button("5", 102, 738, 81, 80)
        self.button_for_number_6 = button.Button("6", 187, 738, 81, 80)
        self.button_for_number_7 = button.Button("7", 18, 823, 81, 80)
        self.button_for_number_8 = button.Button("8", 102, 823, 81, 80)
        self.button_for_number_9 = button.Button("9", 187, 823, 81, 80)

        self.board_surface = pygame.Surface((400, 400), pygame.SRCALPHA)

        self.initialize()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()

                if self.button_for_main_menu.is_clicked(event):
                    self.main_menu.run()

                elif self.button_for_checking.is_clicked(event):
                    print("Checking...")
                elif self.button_for_eraser.is_clicked(event):
                    print("Erasing...")
                elif self.button_for_number_1.is_clicked(event):
                    print("Placing 1...")
                elif self.button_for_number_2.is_clicked(event):
                    print("Placing 2...")
                elif self.button_for_number_3.is_clicked(event):
                    print("Placing 3...")
                elif self.button_for_number_4.is_clicked(event):
                    print("Placing 4...")
                elif self.button_for_number_5.is_clicked(event):
                    print("Placing 5...")
                elif self.button_for_number_6.is_clicked(event):
                    print("Placing 6...")
                elif self.button_for_number_7.is_clicked(event):
                    print("Placing 7...")
                elif self.button_for_number_8.is_clicked(event):
                    print("Placing 8...")
                elif self.button_for_number_9.is_clicked(event):
                    print("Placing 9...")

            self.screen.blit(self.background_image, (0, 0))

            self.draw_board_clear(self.screen, self.board_surface)

            self.button_for_checking.draw(self.screen)
            self.button_for_eraser.draw(self.screen)
            self.button_for_main_menu.draw(self.screen)

            self.button_for_number_1.draw(self.screen)
            self.button_for_number_2.draw(self.screen)
            self.button_for_number_3.draw(self.screen)
            self.button_for_number_4.draw(self.screen)
            self.button_for_number_5.draw(self.screen)
            self.button_for_number_6.draw(self.screen)
            self.button_for_number_7.draw(self.screen)
            self.button_for_number_8.draw(self.screen)
            self.button_for_number_9.draw(self.screen)

            pygame.display.flip()
