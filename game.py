import pygame
import button
import solver_logic

class Game:
    """Main class for the game."""

    def __init__(self, height, width, main_menu, board):

        self.height = height
        self.width = width

        self.solver_logic = solver_logic.SolverLogic
        self.main_menu = main_menu
        self.board = board

        self.background_image = None
        self.screen = None

        self.button_for_pen = None
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

    # This funcion draw the cells
    def draw_highlighted_cells(self, button_instance, surface, board_buttons_instance):
        """Draw a highlighted cell."""
        y, x = button_instance.name
        for row in board_buttons_instance:
            for button_instance_second in row:
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

        # Buttons on the toppom of the screen
        self.button_for_checking = button.Button("check", 318, 124, 76, 76)
        self.button_for_main_menu = button.Button("main_menu", 18, 124, 76, 76)
        self.button_for_hint = button.Button("hint", 168, 124, 76, 76)

        # Buttons on the right for tools
        self.button_for_pen = button.Button("pen", 300, 750, 80, 80)
        self.button_for_eraser = button.Button("eraser", 300, 644, 80, 80)
        self.button_for_notes = button.Button("taking notes", 300, 814, 80, 80)

        # Buttons for numbers
        self.button_for_number_1 = button.Button("1", 24, 644, 81, 80)
        self.button_for_number_2 = button.Button("2", 109, 644, 81, 80)
        self.button_for_number_3 = button.Button("3", 193, 644, 81, 80)

        self.button_for_number_4 = button.Button("4", 24, 728, 81, 80)
        self.button_for_number_5 = button.Button("5", 109, 728, 81, 80)
        self.button_for_number_6 = button.Button("6", 193, 728, 81, 80)

        self.button_for_number_7 = button.Button("7", 24, 814, 81, 80)
        self.button_for_number_8 = button.Button("8", 109, 814, 81, 80)
        self.button_for_number_9 = button.Button("9", 193, 814, 81, 80)
        
    def run(self,muted):
        list_of_buttons_changed = self.solver_logic.create_a_riddle(self.solver_logic, self.board.list_of_buttons)
        self.board.list_of_buttons = list_of_buttons_changed
        self.initialize()

        running = True

        tool = "placing numbers"

        board_column_chosed = None
        position_of_board_column = None
        number_or_hint = None

        while running:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()

                # Funcion for the top buttons
                if self.button_for_main_menu.is_clicked(event):
                    running = False
                    self.main_menu.run(muted)

                elif self.button_for_checking.is_clicked(event):
                    print("Checking...")

                elif self.button_for_hint.is_clicked(event):
                    if board_column_chosed is not None:
                        if tool == "taking notes":
                            tool = "placing numbers"
                        number_or_hint = "hint"
                        print(
                            number_or_hint,
                            position_of_board_column,
                            tool,
                        )
                        number_or_hint = None

                # Funcionality for tool buttons
                elif self.button_for_pen.is_clicked(event):
                    if tool == "taking notes" or "eraser":
                        tool = "placing numbers"
                    else:
                        tool = None

                elif self.button_for_eraser.is_clicked(event):
                    if tool == "taking notes" or "placing numbers":
                        tool = "eraser"
                        if board_column_chosed is not None:
                            board_column_chosed.data = None
                            board_column_chosed.notes = []
                    else:
                        tool = None

                elif self.button_for_notes.is_clicked(event):
                    if tool == "placing numbers" or "eraser":
                        tool = "taking notes"
                    else:
                        tool = None

                # numbers funcioonality
                elif self.button_for_number_1.is_clicked(event):
                    if board_column_chosed is not None:
                        number_or_hint = "1"
                        if tool == "placing numbers":
                            if board_column_chosed.data != 1:
                                board_column_chosed.data = 1
                                board_column_chosed.notes = []
                            elif board_column_chosed.data == 1:
                                board_column_chosed.data = None
                                board_column_chosed.notes = []
                        elif tool == "taking notes":
                            what_to_do = "adding"
                            for number in board_column_chosed.notes:
                                if number == 1:
                                    what_to_do = "removing"
                            if what_to_do == "adding":
                                board_column_chosed.notes.append(1)
                            elif what_to_do == "removing":
                                board_column_chosed.notes.remove(1)
                        elif tool == "eraser":
                            board_column_chosed.data = None
                            board_column_chosed.notes = []
                        number_or_hint = None

                elif self.button_for_number_2.is_clicked(event):
                    if board_column_chosed is not None:
                        number_or_hint = "2"
                        if tool == "placing numbers":
                            if board_column_chosed.data != 2:
                                board_column_chosed.data = 2
                                board_column_chosed.notes = []
                            elif board_column_chosed.data == 2:
                                board_column_chosed.data = None
                                board_column_chosed.notes = []
                        elif tool == "taking notes":
                            what_to_do = "adding"
                            for number in board_column_chosed.notes:
                                if number == 2:
                                    what_to_do = "removing"
                            if what_to_do == "adding":
                                board_column_chosed.notes.append(2)
                            elif what_to_do == "removing":
                                board_column_chosed.notes.remove(2)
                        elif tool == "eraser":
                            board_column_chosed.data = None
                            board_column_chosed.notes = []
                        number_or_hint = None

                elif self.button_for_number_3.is_clicked(event):
                    if board_column_chosed is not None:
                        number_or_hint = "3"
                        if tool == "placing numbers":
                            if board_column_chosed.data != 3:
                                board_column_chosed.data = 3
                                board_column_chosed.notes = []
                            elif board_column_chosed.data == 3:
                                board_column_chosed.data = None
                                board_column_chosed.notes = []
                        elif tool == "taking notes":
                            what_to_do = "adding"
                            for number in board_column_chosed.notes:
                                if number == 3:
                                    what_to_do = "removing"
                            if what_to_do == "adding":
                                board_column_chosed.notes.append(3)
                            elif what_to_do == "removing":
                                board_column_chosed.notes.remove(3)
                        elif tool == "eraser":
                            board_column_chosed.data = None
                            board_column_chosed.notes = []
                        number_or_hint = None

                elif self.button_for_number_4.is_clicked(event):
                    if board_column_chosed is not None:
                        number_or_hint = "4"
                        if tool == "placing numbers":
                            if board_column_chosed.data != 4:
                                board_column_chosed.data = 4
                                board_column_chosed.notes = []
                            elif board_column_chosed.data == 4:
                                board_column_chosed.data = None
                                board_column_chosed.notes = []
                        elif tool == "taking notes":
                            what_to_do = "adding"
                            for number in board_column_chosed.notes:
                                if number == 4:
                                    what_to_do = "removing"
                            if what_to_do == "adding":
                                board_column_chosed.notes.append(4)
                            elif what_to_do == "removing":
                                board_column_chosed.notes.remove(4)
                        elif tool == "eraser":
                            board_column_chosed.data = None
                            board_column_chosed.notes = []
                        number_or_hint = None

                elif self.button_for_number_5.is_clicked(event):
                    if board_column_chosed is not None:
                        number_or_hint = "5"
                        if tool == "placing numbers":
                            if board_column_chosed.data != 5:
                                board_column_chosed.data = 5
                                board_column_chosed.notes = []
                            elif board_column_chosed.data == 5:
                                board_column_chosed.data = None
                                board_column_chosed.notes = []
                        elif tool == "taking notes":
                            what_to_do = "adding"
                            for number in board_column_chosed.notes:
                                if number == 5:
                                    what_to_do = "removing"
                            if what_to_do == "adding":
                                board_column_chosed.notes.append(5)
                            elif what_to_do == "removing":
                                board_column_chosed.notes.remove(5)
                        elif tool == "eraser":
                            board_column_chosed.data = None
                            board_column_chosed.notes = []
                        number_or_hint = None

                elif self.button_for_number_6.is_clicked(event):
                    if board_column_chosed is not None:
                        number_or_hint = "6"
                        if tool == "placing numbers":
                            if board_column_chosed.data != 6:
                                board_column_chosed.data = 6
                                board_column_chosed.notes = []
                            elif board_column_chosed.data == 6:
                                board_column_chosed.data = None
                                board_column_chosed.notes = []
                        elif tool == "taking notes":
                            what_to_do = "adding"
                            for number in board_column_chosed.notes:
                                if number == 6:
                                    what_to_do = "removing"
                            if what_to_do == "adding":
                                board_column_chosed.notes.append(6)
                            elif what_to_do == "removing":
                                board_column_chosed.notes.remove(6)
                        elif tool == "eraser":
                            board_column_chosed.data = None
                            board_column_chosed.notes = []
                        number_or_hint = None

                elif self.button_for_number_7.is_clicked(event):
                    if board_column_chosed is not None:
                        number_or_hint = "7"
                        if tool == "placing numbers":
                            if board_column_chosed.data != 7:
                                board_column_chosed.data = 7
                                board_column_chosed.notes = []
                            elif board_column_chosed.data == 7:
                                board_column_chosed.data = None
                                board_column_chosed.notes = []
                        elif tool == "taking notes":
                            what_to_do = "adding"
                            for number in board_column_chosed.notes:
                                if number == 7:
                                    what_to_do = "removing"
                            if what_to_do == "adding":
                                board_column_chosed.notes.append(7)
                            elif what_to_do == "removing":
                                board_column_chosed.notes.remove(7)
                        elif tool == "eraser":
                            board_column_chosed.data = None
                            board_column_chosed.notes = []
                        number_or_hint = None

                elif self.button_for_number_8.is_clicked(event):
                    if board_column_chosed is not None:
                        number_or_hint = "8"
                        if tool == "placing numbers":
                            if board_column_chosed.data != 8:
                                board_column_chosed.data = 8
                                board_column_chosed.notes = []
                            elif board_column_chosed.data == 8:
                                board_column_chosed.data = None
                                board_column_chosed.notes = []
                        elif tool == "taking notes":
                            what_to_do = "adding"
                            for number in board_column_chosed.notes:
                                if number == 8:
                                    what_to_do = "removing"
                            if what_to_do == "adding":
                                board_column_chosed.notes.append(8)
                            elif what_to_do == "removing":
                                board_column_chosed.notes.remove(8)
                        elif tool == "eraser":
                            board_column_chosed.data = None
                            board_column_chosed.notes = []
                        number_or_hint = None

                elif self.button_for_number_9.is_clicked(event):
                    if board_column_chosed is not None:
                        number_or_hint = "9"
                        if tool == "placing numbers":
                            if board_column_chosed.data != 9:
                                board_column_chosed.data = 9
                                board_column_chosed.notes = []
                            elif board_column_chosed.data == 9:
                                board_column_chosed.data = None
                                board_column_chosed.notes = []
                        elif tool == "taking notes":
                            what_to_do = "adding"
                            for number in board_column_chosed.notes:
                                if number == 9:
                                    what_to_do = "removing"
                            if what_to_do == "adding":
                                board_column_chosed.notes.append(9)
                            elif what_to_do == "removing":
                                board_column_chosed.notes.remove(9)
                        elif tool == "eraser":
                            board_column_chosed.data = None
                            board_column_chosed.notes = []
                        number_or_hint = None

                for row in self.board.list_of_buttons:
                    for button_instance in row:
                        if button_instance.is_clicked(event):
                            if button_instance.locked is True:
                                break
                            else:
                                if tool in ("placing numbers", "taking notes"):
                                    board_column_chosed = button_instance
                                    position_of_board_column = board_column_chosed.name
                                    self.board.append_to_board(
                                        position_of_board_column[1],
                                        position_of_board_column[0],
                                        board_column_chosed,
                                    )
                                elif tool == "eraser":
                                    board_column_chosed = button_instance
                                    position_of_board_column = board_column_chosed.name
                                    self.board.append_to_board(
                                        position_of_board_column[1],
                                        position_of_board_column[0],
                                        board_column_chosed,
                                    )

                                    board_column_chosed.data = None
                                    board_column_chosed.notes = []
                                break

            self.screen.blit(self.background_image, (0, 0))

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

            for row in self.board.list_of_buttons:
                for button_instance in row:
                    button_instance.draw(self.screen)

            if board_column_chosed is not None:
                self.draw_highlighted_cells(
                    board_column_chosed, self.screen, self.board.list_of_buttons
                )


            for row in self.board.list_of_buttons: 
                for button in row: 
                    if button.data in (1,2,3,4,5,6,7,8,9): 
                        self.board.draw_number_on_button_placing_numbers(
                            button, self.screen, button.data
                        )
                    elif button.notes is not None:
                        for note in button.notes:
                            self.board.draw_number_on_button_notes(
                                button, self.screen, note
                            )

            if tool == "taking notes":
                png_image = pygame.image.load("textures/pencil_clicked.png")
                self.screen.blit(png_image, (296, 810))
            elif tool == "placing numbers":
                png_image = pygame.image.load("textures/pen_clicked.png")
                self.screen.blit(png_image, (296, 726))
            elif tool == "eraser":
                png_image = pygame.image.load("textures/eraser_clicked.png")
                self.screen.blit(png_image, (296, 640))

            pygame.display.flip()
