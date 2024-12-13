import pygame
import button
import board


class Game:
    """Main class for the game."""

    def __init__(self, height, width, solver_logic, main_menu, board):

        self.height = height
        self.width = width

        self.solver_logic = solver_logic
        self.main_menu = main_menu
        self.board = board

        self.background_image = None
        self.screen = None

        self.board_surface = None

        self.button_for_checking = None
        self.button_for_eraser = None
        self.button_for_main_menu = None
        self.button_for_notes = None
        self.button_for_hint = None

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

    def draw_highlighted_cells(self, button_instance, surface, board_buttons_instance):
        """Draw a highlighted cell."""
        y, x = button_instance.name

        for button_instance_second in board_buttons_instance:
            if (
                button_instance_second.name[0] == y
                or button_instance_second.name[1] == x
            ):
                button_instance_second.draw(surface)

                temp_surface = pygame.Surface(
                    (
                        button_instance_second.rect.width,
                        button_instance_second.rect.height,
                    ),
                    pygame.SRCALPHA,
                )
                temp_surface.fill((100, 100, 100, 100))

                surface.blit(
                    temp_surface,
                    (button_instance_second.rect.x, button_instance_second.rect.y),
                )

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
        self.button_for_notes = button.Button("taking notes", 7, 135, 76, 76)
        self.button_for_hint = button.Button("hint", 329, 135, 76, 76)

        self.button_for_number_1 = button.Button("1", 18, 653, 81, 80)
        self.button_for_number_2 = button.Button("2", 102, 653, 81, 80)
        self.button_for_number_3 = button.Button("3", 187, 653, 81, 80)
        self.button_for_number_4 = button.Button("4", 18, 738, 81, 80)
        self.button_for_number_5 = button.Button("5", 102, 738, 81, 80)
        self.button_for_number_6 = button.Button("6", 187, 738, 81, 80)
        self.button_for_number_7 = button.Button("7", 18, 823, 81, 80)
        self.button_for_number_8 = button.Button("8", 102, 823, 81, 80)
        self.button_for_number_9 = button.Button("9", 187, 823, 81, 80)

        self.initialize()

        board_buttons = self.board.draw(self.screen)

        running = True

        tool = "placing numbers"

        board_column_chosed = None
        position_of_board_column = None

        number_or_eraser_or_hint_chosed = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()

                if self.button_for_main_menu.is_clicked(event):
                    running = False
                    self.main_menu.run()

                elif self.button_for_checking.is_clicked(event):
                    print("Checking...")

                elif self.button_for_eraser.is_clicked(event):
                    number_or_eraser_or_hint_chosed = "eraser"
                    print(
                        number_or_eraser_or_hint_chosed, position_of_board_column, tool
                    )
                    number_or_eraser_or_hint_chosed = None

                elif self.button_for_notes.is_clicked(event):
                    if tool == "taking notes":
                        tool = "placing numbers"
                    else:
                        tool = "taking notes"

                elif self.button_for_hint.is_clicked(event):
                    if tool == "taking notes":
                        tool = "placing numbers"
                    number_or_eraser_or_hint_chosed = "hint"
                    print(
                        number_or_eraser_or_hint_chosed, position_of_board_column, tool
                    )
                    number_or_eraser_or_hint_chosed = None

                elif self.button_for_number_1.is_clicked(event):
                    number_or_eraser_or_hint_chosed = "1"
                    print(
                        number_or_eraser_or_hint_chosed, position_of_board_column, tool
                    )
                    number_or_eraser_or_hint_chosed = None
                elif self.button_for_number_2.is_clicked(event):
                    number_or_eraser_or_hint_chosed = "2"
                    print(
                        number_or_eraser_or_hint_chosed, position_of_board_column, tool
                    )
                    number_or_eraser_or_hint_chosed = None
                elif self.button_for_number_3.is_clicked(event):
                    number_or_eraser_or_hint_chosed = "3"
                    print(
                        number_or_eraser_or_hint_chosed, position_of_board_column, tool
                    )
                    number_or_eraser_or_hint_chosed = None
                elif self.button_for_number_4.is_clicked(event):
                    number_or_eraser_or_hint_chosed = "4"
                    print(
                        number_or_eraser_or_hint_chosed, position_of_board_column, tool
                    )
                    number_or_eraser_or_hint_chosed = None
                elif self.button_for_number_5.is_clicked(event):
                    number_or_eraser_or_hint_chosed = "5"
                    print(
                        number_or_eraser_or_hint_chosed, position_of_board_column, tool
                    )
                    number_or_eraser_or_hint_chosed = None
                elif self.button_for_number_6.is_clicked(event):
                    number_or_eraser_or_hint_chosed = "6"
                    print(
                        number_or_eraser_or_hint_chosed, position_of_board_column, tool
                    )
                    number_or_eraser_or_hint_chosed = None
                elif self.button_for_number_7.is_clicked(event):
                    number_or_eraser_or_hint_chosed = "7"
                    print(
                        number_or_eraser_or_hint_chosed, position_of_board_column, tool
                    )
                    number_or_eraser_or_hint_chosed = None
                elif self.button_for_number_8.is_clicked(event):
                    number_or_eraser_or_hint_chosed = "8"
                    print(
                        number_or_eraser_or_hint_chosed, position_of_board_column, tool
                    )
                    number_or_eraser_or_hint_chosed = None
                elif self.button_for_number_9.is_clicked(event):
                    number_or_eraser_or_hint_chosed = "9"
                    print(
                        number_or_eraser_or_hint_chosed, position_of_board_column, tool
                    )
                    number_or_eraser_or_hint_chosed = None

                for button_instance in board_buttons:
                    if button_instance.is_clicked(event):
                        board_column_chosed = button_instance
                        position_of_board_column = board_column_chosed.name
                        print(position_of_board_column, tool)
                        break

            self.screen.blit(self.background_image, (0, 0))
            board_buttons = self.board.draw(self.screen)

            self.button_for_checking.draw(self.screen)
            self.button_for_eraser.draw(self.screen)
            self.button_for_main_menu.draw(self.screen)
            self.button_for_notes.draw(self.screen)
            self.button_for_hint.draw(self.screen)

            self.button_for_number_1.draw(self.screen)
            self.button_for_number_2.draw(self.screen)
            self.button_for_number_3.draw(self.screen)
            self.button_for_number_4.draw(self.screen)
            self.button_for_number_5.draw(self.screen)
            self.button_for_number_6.draw(self.screen)
            self.button_for_number_7.draw(self.screen)
            self.button_for_number_8.draw(self.screen)
            self.button_for_number_9.draw(self.screen)

            if tool == "taking notes":
                pygame.draw.circle(self.screen, (255, 0, 0), (80, 140), (14))

            for button_instance in board_buttons:
                button_instance.draw(self.screen)

            if board_column_chosed is not None:
                self.draw_highlighted_cells(
                    board_column_chosed, self.screen, board_buttons
                )

            pygame.display.flip()
