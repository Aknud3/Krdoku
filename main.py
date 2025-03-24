# Importing the kivy framework
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.core.audio import SoundLoader
from kivy.uix.screenmanager import NoTransition
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.resources import resource_find
from kivy.resources import resource_add_path
from kivy.utils import platform

# Importing all the other libraries
import random
import os
import cv2
import numpy as np
from plyer import filechooser
import android
from android.storage import primary_external_storage_path
from jnius import autoclass
from android.permissions import request_permissions, Permission
from android import activity, mActivity


File = autoclass("java.io.File")
Interpreter = autoclass("org.tensorflow.lite.Interpreter")
InterpreterOptions = autoclass("org.tensorflow.lite.Interpreter$Options")
TensorBuffer = autoclass("org.tensorflow.lite.support.tensorbuffer.TensorBuffer")
ByteBuffer = autoclass("java.nio.ByteBuffer")

request_permissions(
    [Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE]
)

# used for transparency of buttons for testing
TRANSPARENT = 0
DESTROY_CELLS = True  # Udělal jsem si tuhle proměnou a podle toho budu řadit jestli mám udělat nový příklad sudoku, nebo tam mam dat ten starý
GREEN = (0.063, 0.486, 0.255, 1)  # Green that I use for locked buttons


# Main logic class for the App
import random
import numpy as np


class SolverLogic:
    def __init__(self):
        # Precompute cell relationships during initialization
        self.peer_cache = {}
        self.name_to_index = {}
        self._precompute_relationships()

    def _precompute_relationships(self):
        """Precompute all cell relationships and indices once"""
        for row in range(9):
            for col in range(9):
                index = row * 9 + col
                self.name_to_index[f"{row},{col}"] = index

                # Calculate peers (row, column, and box)
                peers = set()
                # Row peers
                peers.update(range(row * 9, (row + 1) * 9))
                # Column peers
                peers.update(range(col, 81, 9))
                # Box peers
                start_row, start_col = 3 * (row // 3), 3 * (col // 3)
                for x in range(3):
                    for y in range(3):
                        peers.add((start_row + x) * 9 + (start_col + y))

                peers.discard(index)  # Remove self
                self.peer_cache[index] = peers

    def is_solved(self, list_of_columns):

        if not self.check(list_of_columns):
            return False

        for cell in list_of_columns:
            if not cell.data or cell.data == "" or cell.data == 0:
                return False

        return True

    def check(self, list_of_columns):
        """Optimized check using precomputed relationships"""
        valid_numbers = set(range(1, 10))
        errors = set()
        cells = list_of_columns

        # Reset colors first
        for cell in cells:
            cell.color = (0, 0, 0, 1)

        # Single pass validation
        for index in range(81):
            cell = cells[index]
            value = cell.data

            if not value or value == "" or value == 0:  # Empty cell
                continue

            if value not in valid_numbers:
                errors.add(index)
                continue

            # Check conflicts using precomputed peers
            for peer in self.peer_cache[index]:
                if cells[peer].data == value:
                    errors.add(index)
                    errors.add(peer)

        # Apply errors
        if errors:
            for index in errors:
                cells[index].color = (1, 0, 0, 1)
            return False
        return True

    def validate(self, column, list_of_columns, number):
        """Optimized validation using cache"""
        index = self.name_to_index[column.name]
        return all(
            list_of_columns[peer].data != number for peer in self.peer_cache[index]
        )

    def create_a_riddle(self, list_of_columns, difficulty=20):
        """Create a riddle from the list of columns."""
        for column in list_of_columns:
            if column.data is None:  # Empty cell
                numbers = list(range(1, 10))
                random.shuffle(numbers)  # Randomize numbers to generate unique puzzles

                for number in numbers:
                    if self.validate(column, list_of_columns, number):
                        column.data = number
                        column.locked = True
                        # Recursively attempt to solve the rest of the grid
                        if self.create_a_riddle(list_of_columns, difficulty):
                            return list_of_columns

                        # Backtrack: Undo the tentative placement
                        column.data = None

                # If no number fits, backtrack
                return False

        # If all cells are filled, return the completed list
        cells_to_remove = difficulty  # Number of cells to remove to create a puzzle
        while cells_to_remove > 0:
            index = random.randint(0, 80)
            if list_of_columns[index].data is not None:
                list_of_columns[index].data = ""
                list_of_columns[index].locked = False
                cells_to_remove -= 1

        return list_of_columns

    def solve_a_riddle(self, list_of_columns):
        cells = [
            cell.data if cell.data not in (None, "") else 0 for cell in list_of_columns
        ]

        if self._solve(cells):
            for i, cell in enumerate(list_of_columns):
                cell.data = cells[i] if cells[i] != 0 else ""
            return list_of_columns  # Vracíme seznam vyřešených Column objektů
        return False

    def _solve(self, cells):
        """Interní řešicí metoda pracující s číselným polem"""
        try:
            idx = cells.index(0)
        except ValueError:
            return True  # Všechna pole vyplněna

        row, col = divmod(idx, 9)

        for num in range(1, 10):
            if self._is_valid(cells, row, col, num):
                cells[idx] = num
                if self._solve(cells):
                    return True
                cells[idx] = 0  # Backtrack
        return False

    def _is_valid(self, cells, row, col, num):
        """Validace pro číselné pole"""
        # Kontrola řádku
        if num in cells[row * 9 : (row + 1) * 9]:
            return False

        # Kontrola sloupce
        if num in cells[col::9]:
            return False

        # Kontrola bloku
        start_row = row - row % 3
        start_col = col - col % 3
        for i in range(3):
            for j in range(3):
                if cells[(start_row + i) * 9 + (start_col + j)] == num:
                    return False
        return True


class TensorFlowModel:
    def __init__(self):
        self.interpreter = None
        self.input_shape = None
        self.output_shape = None
        self.output_type = None

    def load(self, model_filename, num_threads=None):
        try:
            model = File(model_filename)
            options = InterpreterOptions()
            if num_threads is not None:
                options.setNumThreads(num_threads)
            self.interpreter = Interpreter(model, options)
            self.allocate_tensors()
            print(f"[DEBUG] Model načten: {model_filename}")
            print(
                f"[DEBUG] Output data type: {self.interpreter.getOutputTensor(0).dataType()}"
            )
        except Exception as e:
            print(f"[ERROR] Chyba při načítání modelu: {e}")
            raise

    def allocate_tensors(self):
        self.interpreter.allocateTensors()
        self.input_shape = list(self.interpreter.getInputTensor(0).shape())
        self.output_shape = list(self.interpreter.getOutputTensor(0).shape())
        self.output_type = self.interpreter.getOutputTensor(0).dataType()

    def get_input_shape(self):
        return self.input_shape[1:]  # Bez batch dimenze

    def resize_input(self, new_shape):
        new_shape = [1] + list(new_shape)
        if self.input_shape != new_shape:
            self.interpreter.resizeInput(0, new_shape)
            self.allocate_tensors()

    def predict(self, x):
        try:
            x = x.astype(np.float32)
            print(f"[DEBUG] Vstupní tvar: {x.shape}")
            if x.shape != tuple(self.input_shape):
                x = np.expand_dims(x, axis=0)
            input_buffer = ByteBuffer.wrap(x.tobytes())
            output_buffer = TensorBuffer.createFixedSize(
                self.output_shape, self.output_type
            )
            self.interpreter.run(input_buffer, output_buffer.getBuffer().rewind())
            output_array = self._convert_output(output_buffer)
            print(f"[DEBUG] Výstupní predikce: {output_array}")
            return output_array[0]  # Odstranění batch dimenze
        except Exception as e:
            print(f"[ERROR] Chyba v predict: {e}")
            raise

    def _convert_output(self, output_buffer):
        type_str = self.output_type.toString()
        print(f"[DEBUG] Output type: {type_str}")
        if "FLOAT32" in type_str:
            return np.array(output_buffer.getFloatArray()).reshape(self.output_shape)
        elif "INT32" in type_str:
            return np.array(output_buffer.getIntArray()).reshape(self.output_shape)
        elif "UINT8" in type_str:
            return np.array(output_buffer.getUint8Array()).reshape(self.output_shape)
        elif "INT64" in type_str:
            return np.array(output_buffer.getLongArray()).reshape(self.output_shape)
        else:
            raise ValueError(f"Unsupported output type: {type_str}")


# Normal button that I use
class CustomButton(Button):
    """Custom Button that uses relative positioning and sizing and is transaparent, so I can use it in my texture based app"""

    def __init__(self, name, rel_x, rel_y, rel_width, rel_height, **kwargs):
        super().__init__(**kwargs)
        self.text = name
        self.size_hint = (rel_width, rel_height)  # Proportional size
        self.pos_hint = {"x": rel_x, "y": rel_y}  # Proportional position

        self.background_color = (1, 1, 1, TRANSPARENT)  # Transparent background
        self.color = (1, 1, 1, TRANSPARENT)  # Text color


# Column class for the sudoku board
class Column(Button):
    def __init__(self, name, rel_x, rel_y, rel_width, rel_height, block, **kwargs):
        super().__init__(**kwargs)

        # Set size and position hints
        self.name = name
        self.text = ""
        self.size_hint = (rel_width, rel_height)
        self.pos_hint = {"x": rel_x, "y": rel_y}

        # Logic properties
        self.data = None
        self.locked = None
        self.notes = [False] * 9  # Active notes for numbers 1-9
        self.block = block

        self.background_color = (0, 0, 0, 0)  # White
        self.color = (0, 0, 0, 1)  # Black

        # Text formatting
        self.font_size = 24
        self.bold = True

        # Create labels for notes (3x3 grid)S
        self.note_labels = []
        for _ in range(9):
            label = Label(
                text="",
                font_size=12,
                bold=False,
                color=(0, 0, 0, 1),  # Black text
                size_hint=(None, None),
                halign="center",
                valign="middle",
            )
            label.bind(size=self._update_label_size)
            self.note_labels.append(label)
            self.add_widget(label)

        # Update the positions of the labels when the size changes
        self.bind(size=self._update_notes_position, pos=self._update_notes_position)

    # Functions for the column
    def _update_label_size(self, instance, size):
        """Ensure the text is centered properly."""
        instance.text_size = instance.size

    def _update_notes_position(self, *args):
        """Position the labels to create a 3x3 grid inside the button."""
        # This just calculates where notes should be displayed on the button
        width, height = self.size
        cell_width = width / 3
        cell_height = height / 2.9

        for i, label in enumerate(self.note_labels):
            col = i % 3
            row = i // 3

            label.size = (cell_width, cell_height)
            label.pos = (self.x + col * cell_width, self.y + (2 - row) * cell_height)

    def display_number(self, number=0):
        """Display a number or notes on the button."""
        if number == "0":  # This is for notes
            self.text = ""
            for i, label in enumerate(self.note_labels):
                label.text = str(i + 1) if self.notes[i] else ""
        else:  # This is for normal numbers
            # Clear notes and display only the given number
            if self.locked:
                self.color = (0.063, 0.486, 0.255, 1)

            self.text = str(number)
            for label in self.note_labels:
                label.text = ""

    def correct_font_size(self):
        """Correct the font size of the number and notes on the button."""
        # This just dynamically changes the font size based on the screen size
        self.font_size = (Window.width * Window.height) / (917 * 412) * 10
        for label in self.note_labels:
            label.font_size = self.font_size / 2


# All Screen Classes
# Main Screens
class MainMenuScreen(Screen):
    def __init__(self, **kwargs):

        # I will reformat this because I will use only one file for the whole app
        super().__init__(**kwargs)
        self.muted = True  # Mute state
        self.layout = FloatLayout()  # Kivy layout

        self.update_background()  # updating background image based on mute state, so this I will make the initioal call to update the background

        # Create buttons
        # Button for starging Hrát Klasické Sudoku
        self.play_button = CustomButton(
            name="Play Game",
            rel_x=0.135,
            rel_y=0.127,
            rel_width=0.730,
            rel_height=0.165,
        )  # It is all proportional to the screen size, I calculated it based on the native resolition and native position and size of button
        self.play_button.bind(on_press=self.play_game)  # Button press handler
        self.layout.add_widget(
            self.play_button
        )  # Adding the button to the layout specific for Kivy based app

        # Button for starting Krřešitel
        self.solve_button = CustomButton(
            name="Solve Puzzle",
            rel_x=0.118,
            rel_y=0.426,
            rel_width=0.730,
            rel_height=0.165,
        )
        self.solve_button.bind(on_press=self.solve_puzzle)  # Button press handler
        self.layout.add_widget(self.solve_button)

        # Button for mute indication
        self.mute_button = CustomButton(
            name="Mute", rel_x=0.766, rel_y=0.007, rel_width=0.2, rel_height=0.09
        )
        self.mute_button.bind(on_press=self.toggle_mute)  # Button press handler
        self.layout.add_widget(self.mute_button)

        # Button for opening my Github repository
        self.github_button = CustomButton(
            name="GitHub", rel_x=0.0388, rel_y=0.007, rel_width=0.2, rel_height=0.09
        )
        self.github_button.bind(on_press=self.open_github)  # Button press handler
        self.layout.add_widget(self.github_button)

        self.add_widget(self.layout)

        self.number_of_enterings_for_game = 0

    # Other important functions
    def update_background(self):
        """Update the background image based on the mute state"""
        if self.muted:
            graphics = "textures/main_menu_graphics_muted.png"
        else:
            graphics = "textures/main_menu_graphics.png"

        if hasattr(self, "background"):
            self.layout.remove_widget(self.background)

        self.background = Image(
            source=graphics,
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),  # this makes the image take the whole screen
            pos_hint={"center_x": 0.5, "center_y": 0.5},  # this centers the image
        )
        self.layout.add_widget(self.background, index=0)

    # Button press handlers and logic
    def play_game(self, instance):
        """Handle 'Play Game' button press"""
        if self.number_of_enterings_for_game == 0:
            self.number_of_enterings_for_game += 1
            self.manager.current = "chose_dificulty"
        else:
            game_screen = self.manager.get_screen("game")
            game_screen.difficulty = 0
            self.manager.current = "game"

    def solve_puzzle(self, instance):
        """Handle 'Solve Puzzle' button press"""
        solver_screen = self.manager.get_screen("solver")
        self.manager.current = "solver"

    def toggle_mute(self, instance):
        """Handle 'Mute' button press"""
        self.muted = not self.muted
        self.update_background()
        if self.muted:
            self.manager.music.stop()
        else:
            self.manager.music.play()

    def open_github(self, instance):
        """Open the GitHub link when the button is pressed"""


class GameScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = FloatLayout()
        self.add_widget(self.layout)
        self.background = Image(
            source="textures/game_graphics_pen.png",
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.layout.add_widget(self.background, index=0)

        # Create list of buttons for every button series
        self.buttons_for_board = []
        self.buttons_on_top = []
        self.buttons_for_numbers = []
        self.buttons_for_tools = []

        self.tool = "pen"

        # Buttons on the top
        for x in (
            (0.032, "MainMenu", "go_to_main_menu"),
            (0.4, "Hint", "give_a_hint"),
            (0.762, "Check", "check_solution"),
        ):  #  x[0] is the relative x position, x[1] is the name of the button, x[2] is the action to be performed when the button is pressed
            button_instance = CustomButton(
                name=x[1], rel_x=x[0], rel_y=0.78, rel_width=0.2, rel_height=0.09
            )

            self.buttons_on_top.append(button_instance)

            button_instance.bind(
                on_press=lambda instance, action=x[2]: getattr(self, action)(instance)
            )  #  This binds the button press to the action specified in the list
            self.layout.add_widget(button_instance)

        # Buttons for the tools, the logic is same as for the button on top
        for x in (
            (0.21, "Eraser", "use_eraser"),
            (0.116, "Pen", "use_pen"),
            (0.024, "Pencil", "use_pencil"),
        ):
            button_intsance = CustomButton(
                name=x[1], rel_x=0.72, rel_y=x[0], rel_width=0.21, rel_height=0.09
            )

            self.buttons_for_tools.append(button_intsance)
            button_intsance.bind(
                on_press=lambda instance, action=x[2]: getattr(self, action)(instance)
            )
            self.layout.add_widget(button_intsance)

        # Button for numbers, still same logic
        for x in (
            (0.055, 0.21, "1"),
            (0.26, 0.21, "2"),
            (0.465, 0.21, "3"),
            (0.055, 0.116, "4"),
            (0.26, 0.116, "5"),
            (0.465, 0.116, "6"),
            (0.055, 0.024, "7"),
            (0.26, 0.024, "8"),
            (0.465, 0.024, "9"),
        ):
            button_instance = CustomButton(
                name=x[2],
                rel_x=x[0],
                rel_y=x[1],
                rel_width=0.2,
                rel_height=0.09,
            )
            self.buttons_for_numbers.append(button_instance)
            button_instance.bind(
                on_press=lambda instance, x=x: self.pick_number(instance, int(x[2]))
            )
            self.layout.add_widget(button_instance)

        # Button for board

        # This loop is kinda a mess, but it is just a way to make the board that is dynamic and work
        offset_y = 0  # This is the offset, that I need because the 4 pixels bold lines
        for row in range(9):
            offset_x = 0
            if row % 3 == 0 and row != 0:
                offset_y += 0.0024261844660194  # offset for the 4 pixels XD
            for column in range(9):
                if column % 3 == 0 and column != 0:
                    offset_x += 0.0043620501635768  # offset for the 4 pixels XD

                # Here I calculate in what 3x3 I am, I will need it later
                block_row = row // 3
                block_col = column // 3
                b = block_row * 3 + block_col

                # This creates the button
                self.buttons_for_board.append(
                    Column(
                        name=f"{row},{column}",
                        rel_x=0.0145
                        + offset_x
                        + 0.1067 * column,  # I need it like this
                        rel_y=0.7085 - offset_y - 0.04798 * row,
                        rel_width=0.1067,
                        rel_height=0.0483,
                        block=f"{b}",
                    )
                )
                button = self.buttons_for_board[row * 9 + column]
                button.bind(
                    on_press=lambda instance, btn=button: self.pick_square(
                        instance, btn
                    )
                )

                self.layout.add_widget(
                    self.buttons_for_board[row * 9 + column]
                )  # Add the button to the layout

        # This ends the initialization of the game
        self.solver_logic = SolverLogic()
        self.difficulty = None

    # Functions that handle buttons that all do specific things
    def go_to_main_menu(self, instance):
        """Handle 'Main Menu' button press"""
        self.manager.current = "main_menu"

    def give_a_hint(self, instance):
        """Dá nápovědu pro vybranou buňku: pokud deska neobsahuje konflikty,
        vyřeší sudoku a doplní do vybrané buňky správné číslo, které následně uzamkne.
        """
        # Pokud není vybrána žádná buňka, ukončíme funkci
        if not hasattr(self, "picked_column") or self.picked_column is None:
            return

        # Pomocná funkce pro kontrolu konfliktů, ignoruje prázdné buňky
        def board_has_conflicts(list_of_columns):
            """
            Zkontroluje správnost sudoku desky:
            - Kontroluje řádky, sloupce i 3x3 boxy.
            - Prázdné buňky (None nebo "") ignoruje.
            - Pokud je hodnota neplatná (není 1-9) nebo se objeví duplicita, buňka se označí červeně.

            Vrací:
                True  - pokud jsou všechny ne-prázdné hodnoty v pořádku,
                False - pokud je nalezena chyba.
            """
            correct = True
            valid_numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9}

            # --- Kontrola řádků ---
            for i in range(9):
                row = list_of_columns[i * 9 : (i + 1) * 9]
                seen = {}
                for cell in row:
                    if cell.data in (None, ""):
                        correct = False
                        continue
                    if cell.data not in valid_numbers:
                        cell.color = (1, 0, 0, 1)
                        correct = False
                    else:
                        if cell.data in seen:
                            seen[cell.data].append(cell)
                        else:
                            seen[cell.data] = [cell]
                for cells in seen.values():
                    if len(cells) > 1:
                        for cell in cells:
                            cell.color = (1, 0, 0, 1)
                        correct = False

            # --- Kontrola sloupců ---
            for j in range(9):
                column = [list_of_columns[i * 9 + j] for i in range(9)]
                seen = {}
                for cell in column:
                    if cell.data in (None, ""):
                        correct = False
                        continue
                    if cell.data not in valid_numbers:
                        cell.color = (1, 0, 0, 1)
                        correct = False
                    else:
                        if cell.data in seen:
                            seen[cell.data].append(cell)
                        else:
                            seen[cell.data] = [cell]
                for cells in seen.values():
                    if len(cells) > 1:
                        for cell in cells:
                            cell.color = (1, 0, 0, 1)
                        correct = False

            # --- Kontrola 3x3 boxů ---
            for box_row in range(3):
                for box_col in range(3):
                    box_cells = []
                    for i in range(box_row * 3, box_row * 3 + 3):
                        for j in range(box_col * 3, box_col * 3 + 3):
                            box_cells.append(list_of_columns[i * 9 + j])
                    seen = {}
                    for cell in box_cells:
                        if cell.data in (None, ""):
                            correct = False
                            continue
                        if cell.data not in valid_numbers:
                            cell.color = (1, 0, 0, 1)
                            correct = False
                        else:
                            if cell.data in seen:
                                seen[cell.data].append(cell)
                            else:
                                seen[cell.data] = [cell]
                    for cells in seen.values():
                        if len(cells) > 1:
                            for cell in cells:
                                cell.color = (1, 0, 0, 1)
                            correct = False

            return True

        # Pokud aktuální deska obsahuje konflikty, nápovědu neposkytujeme
        if not board_has_conflicts(self.buttons_for_board):
            return

        # Vytvoříme kopii aktuálního stavu sudoku
        board_copy = []
        for cell in self.buttons_for_board:
            new_cell = Column(
                cell.name,
                cell.pos_hint["x"],
                cell.pos_hint["y"],
                cell.size_hint[0],
                cell.size_hint[1],
                cell.block,
            )
            new_cell.data = cell.data
            new_cell.locked = cell.locked
            board_copy.append(new_cell)

        # Pokusíme se vyřešit sudoku pomocí backtrackingu
        solution = self.solver_logic.solve_a_riddle(board_copy)
        if not solution:
            return

        # Pokud řešení obsahuje 81 prvků (flat list), převedeme jej na 2D formát
        if (
            isinstance(solution, list)
            and len(solution) == 81
            and isinstance(solution[0], Column)
        ):
            solution = [solution[i * 9 : (i + 1) * 9] for i in range(9)]

        # Získáme souřadnice vybrané buňky (formát "řádek,sloupec")
        row_idx, col_idx = map(int, self.picked_column.name.split(","))
        correct_digit = solution[row_idx][col_idx].data

        # Vyplníme vybranou buňku správným číslem a uzamkneme ji
        self.picked_column.data = correct_digit
        self.picked_column.display_number(correct_digit)
        self.picked_column.locked = True

    def check_solution(self, instance):
        """Handle 'Check' button press"""
        self.solver_logic.check(self.buttons_for_board)
        if self.solver_logic.is_solved(self.buttons_for_board):
            main_menu_screen = self.manager.get_screen("main_menu")
            main_menu_screen.number_of_enterings_for_game = 0

            self.manager.current = "correct_solution"

        else:
            self.manager.current = "incorrect_solution"

    def use_eraser(self, instance):
        """Handle 'Eraser' button press"""
        self.tool = "eraser"
        if self.picked_column is not None:
            if self.picked_column.data in (1, 2, 3, 4, 5, 6, 7, 8, 9):
                self.picked_column.data = ""
                self.picked_column.color = (0, 0, 0, 1)
                if self.picked_column.notes != [False] * 9:
                    self.picked_column.display_number("0")
                else:
                    self.picked_column.display_number(self.picked_column.data)
            else:
                self.picked_column.notes = [False] * 9
                self.picked_column.display_number("0")

        self.update_background(self)

    def use_pen(self, instance):
        """Handle 'Pen' button press"""
        self.tool = "pen"
        self.update_background(self)

    def use_pencil(self, instance):
        """Handle 'Pencil' button press"""
        self.tool = "pencil"
        self.update_background(self)

    # Functions that handle number choice and square choice
    def pick_number(self, instance, number):
        """Handle number button press"""
        # This function is like working it is kinda self explanatory a lot of ifs, elifs and else
        self.picked_number = number
        if self.tool == "pen":
            if self.picked_column is not None:
                row, col = self.picked_column.name.split(",")
                row, col = int(row), int(col)
                button = self.buttons_for_board[row * 9 + col]
                if button.data == number:
                    if button.notes != [False] * 9:
                        button.data = "0"
                    else:
                        button.data = ""
                else:
                    button.data = number
                button.display_number(button.data)

        elif self.tool == "pencil":
            if self.picked_column is not None:
                row, col = self.picked_column.name.split(",")
                row, col = int(row), int(col)
                button = self.buttons_for_board[row * 9 + col]
                button.notes[number - 1] = not button.notes[number - 1]
                button.display_number("0")

    def pick_square(self, instance, column):
        """Handle square button press"""
        if column.locked:
            return False

        if self.tool == "eraser":
            column.data = ""
            column.display_number(column.data)

        self.picked_column = column

        self.highlight_square(self)

    # Other important functions
    def correct_font_size(self, *args):
        """Correct the font size of the number on the button"""
        # Dynamically adjust the font size
        for button in self.buttons_for_board:
            button.correct_font_size()

    def run_game(self):
        """Run the game"""
        self.picked_number = None  # Reset the picked number
        self.picked_column = None  # Reset the picked column

        # The App is running in 30 FPS and update is called every frame
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        """Update function called every frame"""
        # Things I need to update every frame, can be done in a better way, but this is the easiest way
        # If i will have problems with memory or performance I will change it

    def on_enter(self):

        global DESTROY_CELLS

        if DESTROY_CELLS:
            for button in self.buttons_for_board:
                button.locked = False
                button.data = None
                button.color = (0, 0, 0, 1)

            dificulty_screen = self.manager.get_screen("chose_dificulty")
            difficulty_from_menu = dificulty_screen.difficulty

            self.buttons_for_board = self.solver_logic.create_a_riddle(
                self.buttons_for_board, difficulty_from_menu
            )

        for button_instance in self.buttons_for_board:
            button_instance.display_number(button_instance.data)

        DESTROY_CELLS = False

        self.run_game()

    def highlight_square(self, instance):
        """Highlight the selected square"""

        if self.picked_column is not None:
            for button in self.buttons_for_board:
                row, col = button.name.split(",")
                row_column, col_column = self.picked_column.name.split(",")
                if (
                    self.picked_column.block == button.block
                    or row == row_column
                    or col == col_column
                ):  # if the button is in the same block, row or column as the picked column it will heightlight it
                    button.background_color = (1, 1, 1, 0.4)
                else:
                    button.background_color = (1, 1, 1, 0)

    def update_background(self, instance):
        """Update the background image based on the mute state"""
        # This is the first thing that come to my mind how to indicate the tools
        if self.tool == "pen":
            graphics = "textures/game_graphics_pen.png"

        elif self.tool == "pencil":
            graphics = "textures/game_graphics_pencil.png"

        elif self.tool == "eraser":
            graphics = "textures/game_graphics_eraser.png"

        if hasattr(self, "background"):
            self.layout.remove_widget(self.background)

        self.background = Image(
            source=graphics,
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.layout.add_widget(self.background, index=len(self.layout.children))


class SolverScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = FloatLayout()
        self.add_widget(self.layout)
        self.background = Image(
            source="textures/solver_graphics.png",
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.model = TensorFlowModel()
        resource_add_path(os.path.dirname(__file__))
        model_path = resource_find("sudoku_model.tflite")
        if model_path:
            self.model.load(model_path)
        else:
            raise FileNotFoundError("Model file not found")

        self.layout.add_widget(self.background, index=0)

        self.button_for_main_menu = CustomButton(
            name="Main Menu", rel_x=0.5785, rel_y=0.04, rel_width=0.345, rel_height=0.16
        )
        self.button_for_main_menu.bind(on_press=self.go_to_main_menu)
        self.layout.add_widget(self.button_for_main_menu)

        self.button_for_taking_a_photo = CustomButton(
            name="Take a photo",
            rel_x=0.088,
            rel_y=0.04,
            rel_width=0.345,
            rel_height=0.16,
        )
        self.button_for_taking_a_photo.bind(on_press=self.photo)
        self.layout.add_widget(self.button_for_taking_a_photo)

        self.temp_image_path = None

    def order_points(self, pts):
        """Seřazení 4 vrcholů obdélníku (pro warp)."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def extract_sudoku_digits(self, image_path):
        print(f"[DEBUG] Začínám zpracovávat obrázek: {image_path}")
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] cv2.imread vrátil None pro cestu: {image_path}")
                return None

            print(f"[DEBUG] Rozměry obrázku: {image.shape}")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            th_inv = cv2.bitwise_not(th)

            contours, _ = cv2.findContours(
                th_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            biggest = None
            max_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000:
                    approx = cv2.approxPolyDP(
                        cnt, 0.02 * cv2.arcLength(cnt, True), True
                    )
                    if len(approx) == 4 and area > max_area:
                        biggest = approx
                        max_area = area

            contour_debug = image.copy()
            if biggest is not None:
                cv2.drawContours(th_inv, [biggest], -1, (0, 255, 0), 2)

            if biggest is None:
                return None

            img_area = image.shape[0] * image.shape[1]
            if max_area < 0.2 * img_area:
                print(
                    f"[ERROR] Největší kontura je příliš malá ({max_area}/{img_area}) - není to sudoku!"
                )
                return None

            # Kontrola poměru stran
            rect = self.order_points(biggest.reshape(4, 2))
            width = np.linalg.norm(rect[1] - rect[0])
            height = np.linalg.norm(rect[3] - rect[0])
            aspect_ratio = width / height
            if not (0.8 < aspect_ratio < 1.2):  # Sudoku je téměř čtverec
                print(f"[ERROR] Špatný poměr stran: {aspect_ratio}")
                return None

            pts1 = np.float32(self.order_points(biggest.reshape(4, 2)))
            side = 450
            pts2 = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
            M = cv2.getPerspectiveTransform(pts1, pts2)

            warped_color = cv2.warpPerspective(th_inv, M, (side, side))

            # Detekce mřížky pomocí Houghovy transformace
            edges = cv2.Canny(warped_color, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(
                edges, 1, np.pi / 180, 80
            )  # Prah 100, ladit podle potřeby

            if (
                lines is None or len(lines) < 16
            ):  # Potřebujeme alespoň 8+8 přímek (9+1 pro okraje)
                print(
                    f"[ERROR] Nedostatek přímek pro mřížku sudoku: {lines.shape if lines is not None else 'žádné'}"
                )
                return None

            horizontal_lines = []
            vertical_lines = []
            for rho, theta in lines[:, 0]:
                if abs(theta) < np.pi / 4 or abs(theta - np.pi) < np.pi / 4:
                    vertical_lines.append((rho, theta))
                elif abs(theta - np.pi / 2) < np.pi / 4:
                    horizontal_lines.append((rho, theta))

            if len(horizontal_lines) < 8 or len(vertical_lines) < 8:
                print(
                    f"[ERROR] Nedostatek horizontálních ({len(horizontal_lines)}) nebo vertikálních ({len(vertical_lines)}) přímek!"
                )
                return None

            print(
                f"[DEBUG] Detekováno {len(horizontal_lines)} horizontálních a {len(vertical_lines)} vertikálních přímek"
            )

            cell_size = side // 9
            for i in range(0, side + 1, cell_size):

                cv2.line(warped_color, (i, 0), (i, side), 255, 2)

                cv2.line(warped_color, (0, i), (side, i), 255, 2)

            blank_grid_inv = cv2.bitwise_not(warped_color)
            warped_gray_inv = cv2.bitwise_not(warped_color)

            sudoku_no_grid = cv2.bitwise_and(warped_gray_inv, blank_grid_inv)
            sudoku_final = cv2.bitwise_not(sudoku_no_grid)

            border_cut = 8

            # Nastavení horního a dolního okraje na černou
            sudoku_final[:border_cut, :] = 0  # Horní okraj
            sudoku_final[-border_cut:, :] = 0  # Dolní okraj

            # Nastavení levého a pravého okraje na černou
            sudoku_final[:, :border_cut] = 0  # Levý okraj
            sudoku_final[:, -border_cut:] = 0  # Pravý okraj

            cell_size = side // 9
            sudoku_digits = [[0] * 9 for _ in range(9)]

            for row in range(9):
                for col in range(9):
                    x1 = col * cell_size
                    y1 = row * cell_size
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size

                    # Výřez buňky
                    cell = sudoku_final[y1:y2, x1:x2]

                    border_cut = 8

                    # Nastavení horního a dolního okraje na černou
                    cell[:border_cut, :] = 0  # Horní okraj
                    cell[-border_cut:, :] = 0  # Dolní okraj

                    # Nastavení levého a pravého okraje na černou
                    cell[:, :border_cut] = 0  # Levý okraj
                    cell[:, -border_cut:] = 0  # Pravý okraj

                    # Resize na 28x28 a normalizace
                    cell_resized = cv2.resize(cell, (28, 28))
                    cell_resized = cell_resized.astype("float32") / 255.0
                    cell_resized = np.expand_dims(
                        cell_resized, axis=-1
                    )  # (28,28) -> (28,28,1)
                    cell_resized = np.expand_dims(
                        cell_resized, axis=0
                    )  # -> (1,28,28,1)

                    # Zobrazit normalizovanou buňku, co jde do modelu

                    mean_val = np.mean(cell)  # Průměrná intenzita pixelů

                    if mean_val < 10:  # Heuristická prahová hodnota pro prázdnou buňku
                        sudoku_digits[row][col] = 0
                        continue

                    # Předpověď
                    pred = self.model.predict(
                        cell_resized
                    )  # Použije novou predict metodu
                    digit = np.argmax(pred)
                    sudoku_digits[row][col] = digit

            if all(all(cell == 9 for cell in row) for row in sudoku_digits):
                print("[ERROR] Všechny buňky jsou prázdné - není to sudoku!")
                return None

            return sudoku_digits

        except Exception as e:
            print(f"[ERROR] Chyba v extract_sudoku_digits: {str(e)}")
            return None

    def go_to_main_menu(self, instance):
        """Handle 'Main Menu' button press"""
        self.manager.current = "main_menu"

    def photo(self, instance):
        if platform == "android":
            Intent = autoclass("android.content.Intent")
            self.intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
            self.intent.addCategory(Intent.CATEGORY_OPENABLE)
            self.intent.setType("image/*")
            mActivity.startActivityForResult(self.intent, 101)
        else:
            filechooser.open_file(
                on_selection=self.file_selection_handler,
                filters=[("Images", "*.png", "*.jpg", "*.jpeg")],
                multiple=False,
            )

    def process_android_image(self, uri):
        try:
            # Get Android context and external directory
            Context = autoclass("android.content.Context")
            context = mActivity.getApplicationContext()
            external_dir = context.getExternalFilesDir(None).getAbsolutePath()
            print(f"[DEBUG] External dir: {external_dir}")

            # Get ContentResolver from context
            resolver = context.getContentResolver()

            # Create a temporary directory for better organization
            temp_dir = os.path.join(external_dir, "temp_images")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # Define the temporary image path
            self.temp_image_path = os.path.join(temp_dir, "temp_sudoku.jpg")
            print(f"[DEBUG] Temp image path: {self.temp_image_path}")

            # Write the image data from the URI to the temporary file
            try:
                with open(self.temp_image_path, "wb") as f:
                    input_stream = resolver.openInputStream(uri)
                    total_bytes = 0
                    buffer = bytearray(1024)

                    while True:
                        count = input_stream.read(buffer)
                        if count <= 0:
                            break
                        f.write(buffer[:count])
                        total_bytes += count

                    print(f"[DEBUG] Successfully saved {total_bytes} bytes")

                # Check if the file is empty and clean up if it is
                if os.path.getsize(self.temp_image_path) == 0:
                    print("[ERROR] File is empty!")
                    if os.path.exists(self.temp_image_path):
                        os.remove(self.temp_image_path)
                    return

            except Exception as e:
                print(f"[ERROR] Error writing file: {str(e)}")
                if os.path.exists(self.temp_image_path):
                    os.remove(self.temp_image_path)
                return

            # Pass the file to the handler
            self.file_selection_handler([self.temp_image_path])

        except Exception as e:
            print(f"[ERROR] Error processing image: {e}")

    def intent_callback(self, requestCode, resultCode, intent):
        if requestCode == 101 and resultCode == -1:  # RESULT_OK
            uri = intent.getData()
            self.process_android_image(uri)

    def file_selection_handler(self, selection):
        if selection:
            photo_path = selection[0]
            print(f"Vybraný obrázek: {photo_path}")

            # Zkontrolujte existenci souboru
            if not os.path.exists(photo_path):
                print("Soubor neexistuje!")
                return

            try:
                sudoku = self.extract_sudoku_digits(photo_path)
            except Exception as e:
                print(f"Chyba při zpracování Sudoku: {e}")
                Clock.schedule_once(lambda dt: self._set_screen("sudoku_not_loaded"), 0)
                return

            if sudoku == [[9] * 9 for _ in range(9)] or sudoku is None:
                Clock.schedule_once(lambda dt: self._set_screen("sudoku_not_loaded"), 0)
            else:
                Clock.schedule_once(lambda dt: self._update_and_switch(sudoku), 0)

            # Úklid dočasných souborů
            if platform == "android" and self.temp_image_path:
                try:
                    os.remove(self.temp_image_path)
                except:
                    pass

    def _set_screen(self, screen_name):
        self.manager.current = screen_name

    def _update_and_switch(self, sudoku):
        editing_screen = self.manager.get_screen("editing_screen")
        editing_screen.update_sudoku(sudoku)
        self.manager.current = "editing_screen"


# Sub Screens for GameScreen
class ChoseDificultyScreen(Screen):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.layout = FloatLayout()
        self.background = Image(
            source="textures/chose_dificulty_screen.png",
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.layout.add_widget(self.background)
        self.add_widget(self.layout)

        self.easy_button = CustomButton(
            name="easy",
            rel_x=0.1359223,
            rel_y=0.64013086,
            rel_width=0.7451456,
            rel_height=0.163576,
        )

        self.easy_button.bind(on_press=self.easy)
        self.layout.add_widget(self.easy_button)

        self.medium_button = CustomButton(
            name="medium",
            rel_x=0.1359223,
            rel_y=0.366412,
            rel_width=0.7451456,
            rel_height=0.163576,
        )

        self.medium_button.bind(on_press=self.medium)
        self.layout.add_widget(self.medium_button)

        self.hard_button = CustomButton(
            name="hard",
            rel_x=0.1359223,
            rel_y=0.09487459105,
            rel_width=0.7451456,
            rel_height=0.163576,
        )

        self.hard_button.bind(on_press=self.hard)
        self.layout.add_widget(self.hard_button)

        self.difficulty = None

    def easy(self, instance):
        # Access the existing instance of the game screen
        self.difficulty = 10
        self.manager.current = "game"

    def medium(self, instance):
        # Access the existing instance of the game screen
        self.difficulty = 20
        self.manager.current = "game"

    def hard(self, instance):
        # Access the existing instance of the game screen
        self.difficulty = 40
        self.manager.current = "game"


class IncorrectSolutionScreen(Screen):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.layout = FloatLayout()
        self.background = Image(
            source="textures/wrong_solution_graphics.png",
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.layout.add_widget(self.background)
        self.add_widget(self.layout)

        self.retry_button = CustomButton(
            name="Retry",
            rel_x=0.109223,
            rel_y=0.08,
            rel_width=0.339805825,
            rel_height=0.152671,
        )
        self.retry_button.bind(on_press=self.retry)
        self.layout.add_widget(self.retry_button)

        self.main_menu_button = CustomButton(
            name="Main Menu",
            rel_x=0.54854368,
            rel_y=0.08,
            rel_width=0.339805825,
            rel_height=0.152671,
        )
        self.main_menu_button.bind(on_press=self.main_menu)
        self.layout.add_widget(self.main_menu_button)

    def retry(self, instance):
        game_screen = self.manager.get_screen("game")
        game_screen.difficulty = 0
        self.manager.current = "game"

    def main_menu(self, instance):
        self.manager.current = "main_menu"


class CorrectSolutionScreen(Screen):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.layout = FloatLayout()
        self.background = Image(
            source="textures/correct_solution_graphics.png",
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.layout.add_widget(self.background)
        self.add_widget(self.layout)

        self.main_menu_button = CustomButton(
            name="Main Menu",
            rel_x=0.33009708,
            rel_y=0.08,
            rel_width=0.339805825,
            rel_height=0.152671,
        )
        self.main_menu_button.bind(on_press=self.main_menu)
        self.layout.add_widget(self.main_menu_button)

    def main_menu(self, instance):
        global DESTROY_CELLS

        DESTROY_CELLS = True
        self.manager.current = "main_menu"


# Sub Screens for Solver Screen
class EditingScreen(Screen):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.layout = FloatLayout()
        self.add_widget(self.layout)
        self.background = Image(
            source="textures/editing.png",
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.layout.add_widget(self.background)

        self.sudoku_data = None
        self.buttons_for_numbers = []
        self.buttons_for_board = []
        self.tool = "pen"

        self.button_for_pen = CustomButton(
            name="pen",
            rel_x=0.7281553,
            rel_y=0.05670665,
            rel_width=0.1941747,
            rel_height=0.0872410,
        )
        self.button_for_pen.bind(on_press=self.use_pen)
        self.layout.add_widget(self.button_for_pen)

        self.button_for_eraser = CustomButton(
            name="eraser",
            rel_x=0.7281553,
            rel_y=0.1504907,
            rel_width=0.1941747,
            rel_height=0.0872410,
        )
        self.button_for_eraser.bind(on_press=self.use_eraser)
        self.layout.add_widget(self.button_for_eraser)

        self.button_for_next = CustomButton(
            name="next",
            rel_x=0.4029126,
            rel_y=0.7818974,
            rel_width=0.1941747,
            rel_height=0.0872410,
        )
        self.button_for_next.bind(on_press=self.next)
        self.layout.add_widget(self.button_for_next)

        for x in (
            (0.055, 0.21, "1"),
            (0.26, 0.21, "2"),
            (0.465, 0.21, "3"),
            (0.055, 0.116, "4"),
            (0.26, 0.116, "5"),
            (0.465, 0.116, "6"),
            (0.055, 0.024, "7"),
            (0.26, 0.024, "8"),
            (0.465, 0.024, "9"),
        ):
            button_instance = CustomButton(
                name=x[2],
                rel_x=x[0],
                rel_y=x[1],
                rel_width=0.2,
                rel_height=0.09,
            )
            self.buttons_for_numbers.append(button_instance)
            button_instance.bind(
                on_press=lambda instance, x=x: self.pick_number(instance, int(x[2]))
            )
            self.layout.add_widget(button_instance)

        self.solver = SolverLogic()

    def next(self, instance):
        self.buttons_for_board = self.solver.solve_a_riddle(self.buttons_for_board)
        if self.buttons_for_board is not False:
            sudoku_solved_screen = self.manager.get_screen("sudoku_solved")
            sudoku_data = self.buttons_for_board
            sudoku_solved_screen.display_sudoku(sudoku_data)
            self.manager.current = "sudoku_solved"

        else:
            self.manager.current = "sudoku_not_solvable"

    def update_sudoku(self, sudoku_data):
        """Update the Sudoku grid with new data"""
        for button in self.buttons_for_board:
            self.layout.remove_widget(button)

        self.buttons_for_board = []

        self.sudoku_data = sudoku_data

        # This loop is kinda a mess, but it is just a way to make the board that is dynamic and work
        offset_y = 0  # This is the offset, that I need because the 4 pixels bold lines
        for row in range(9):
            offset_x = 0
            if row % 3 == 0 and row != 0:
                offset_y += 0.0024261844660194  # offset for the 4 pixels XD
            for column in range(9):
                if column % 3 == 0 and column != 0:
                    offset_x += 0.0043620501635768  # offset for the 4 pixels XD

                # Here I calculate in what 3x3 I am, I will need it later
                block_row = row // 3
                block_col = column // 3
                b = block_row * 3 + block_col

                data_from_photo = self.sudoku_data[row][column]

                # This creates the button
                self.buttons_for_board.append(
                    Column(
                        name=f"{row},{column}",
                        rel_x=0.0145
                        + offset_x
                        + 0.1067 * column,  # I need it like this
                        rel_y=0.7085 - offset_y - 0.04798 * row,
                        rel_width=0.1067,
                        rel_height=0.0483,
                        block=f"{b}",
                    )
                )
                button = self.buttons_for_board[row * 9 + column]
                button.data = data_from_photo
                button.bind(
                    on_press=lambda instance, btn=button: self.pick_square(
                        instance, btn
                    )
                )

                self.layout.add_widget(self.buttons_for_board[row * 9 + column])

        for button in self.buttons_for_board:
            if button.data == 0:
                button.data = ""
            button.display_number(button.data)
            button.correct_font_size()

        self.run_game()

    def run_game(self):
        """Run the game"""
        self.picked_number = None  # Reset the picked number
        self.picked_column = None  # Reset the picked column

        # The App is running in 30 FPS and update is called every frame
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def pick_number(self, instance, number):
        """Handle number button press"""
        # This function is like working it is kinda self explanatory a lot of ifs, elifs and else
        self.picked_number = number
        if self.tool == "pen":
            if self.picked_column is not None:
                row, col = self.picked_column.name.split(",")
                row, col = int(row), int(col)
                button = self.buttons_for_board[row * 9 + col]
                if button.data == number:
                    if button.notes != [False] * 9:
                        button.data = "0"
                    else:
                        button.data = ""
                else:
                    button.data = number
                button.display_number(button.data)

        elif self.tool == "pencil":
            if self.picked_column is not None:
                row, col = self.picked_column.name.split(",")
                row, col = int(row), int(col)
                button = self.buttons_for_board[row * 9 + col]
                button.notes[number - 1] = not button.notes[number - 1]
                button.display_number("0")

    def pick_square(self, instance, column):
        """Handle square button press"""
        if column.locked:
            return False

        if self.tool == "eraser":
            column.data = ""
            column.display_number(column.data)

        self.picked_column = column

        self.highlight_square(self)

    def pick_number(self, instance, number):
        """Handle number button press"""
        # This function is like working it is kinda self explanatory a lot of ifs, elifs and else
        self.picked_number = number
        if self.tool == "pen":
            if self.picked_column is not None:
                row, col = self.picked_column.name.split(",")
                row, col = int(row), int(col)
                button = self.buttons_for_board[row * 9 + col]
                if button.data == number:
                    if button.notes != [False] * 9:
                        button.data = "0"
                    else:
                        button.data = ""
                else:
                    button.data = number
                button.display_number(button.data)

        elif self.tool == "pencil":
            if self.picked_column is not None:
                row, col = self.picked_column.name.split(",")
                row, col = int(row), int(col)
                button = self.buttons_for_board[row * 9 + col]
                button.notes[number - 1] = not button.notes[number - 1]
                button.display_number("0")

    def use_eraser(self, instance):
        """Handle 'Eraser' button press"""
        self.tool = "eraser"
        if self.picked_column is not None:
            if self.picked_column.data in (1, 2, 3, 4, 5, 6, 7, 8, 9):
                self.picked_column.data = ""
                if self.picked_column.notes != [False] * 9:
                    self.picked_column.display_number("0")
                else:
                    self.picked_column.display_number(self.picked_column.data)
            else:
                self.picked_column.notes = [False] * 9
                self.picked_column.display_number("0")

        self.update_background(self)

    def use_pen(self, instance):
        """Handle 'Pen' button press"""
        self.tool = "pen"
        self.update_background(self)

    def highlight_square(self, instance):
        """Highlight the selected square"""

        if self.picked_column is not None:
            for button in self.buttons_for_board:
                row, col = button.name.split(",")
                row_column, col_column = self.picked_column.name.split(",")
                if (
                    self.picked_column.block == button.block
                    or row == row_column
                    or col == col_column
                ):  # if the button is in the same block, row or column as the picked column it will heightlight it
                    button.background_color = (1, 1, 1, 0.4)
                else:
                    button.background_color = (1, 1, 1, 0)

    def update_background(self, instance):
        """Update the background image based on the mute state"""
        # This is the first thing that come to my mind how to indicate the tools
        if self.tool == "pen":
            graphics = "textures/editing.png"

        elif self.tool == "eraser":
            graphics = "textures/editing_eraser_graphics.png"

        if hasattr(self, "background"):
            self.layout.remove_widget(self.background)

        self.background = Image(
            source=graphics,
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.layout.add_widget(self.background, index=len(self.layout.children))

    def correct_font_size(self, *args):
        """Correct the font size of the number on the button"""
        # Dynamically adjust the font size
        for button in self.buttons_for_board:
            button.correct_font_size()

    def update(self, dt):
        """Update function called every frame"""
        # Things I need to update every frame, can be done in a better way, but this is the easiest way
        # If i will have problems with memory or performance I will change it


class SudokuNotLoaded(Screen):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.layout = FloatLayout()
        self.background = Image(
            source="textures/sudoku_not_loaded_graphics.png",
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.layout.add_widget(self.background)
        self.add_widget(self.layout)

        self.main_menu_button = CustomButton(
            name="main_menu",
            rel_x=0.3300970,
            rel_y=0.0305343,
            rel_width=0.3398058,
            rel_height=0.15267175,
        )
        self.main_menu_button.bind(on_press=self.main_menu)
        self.layout.add_widget(self.main_menu_button)

    def main_menu(self, instance):
        self.manager.current = "main_menu"


class SudokuNotSolvable(Screen):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.layout = FloatLayout()
        self.background = Image(
            source="textures/sudoku_not_solvable.png",
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.layout.add_widget(self.background)
        self.add_widget(self.layout)

        self.main_menu_button = CustomButton(
            name="main_menu",
            rel_x=0.3300970,
            rel_y=0.0305343,
            rel_width=0.3398058,
            rel_height=0.15267175,
        )
        self.main_menu_button.bind(on_press=self.main_menu)
        self.layout.add_widget(self.main_menu_button)

    def main_menu(self, instance):
        self.manager.current = "main_menu"


class SudokuSolvedScreen(Screen):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.layout = FloatLayout()
        self.background = Image(
            source="textures/sudoku_display_graphics.png",
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.layout.add_widget(self.background)
        self.add_widget(self.layout)

        self.main_menu_button = CustomButton(
            name="main_menu",
            rel_x=0.3300970,
            rel_y=0.0305343,
            rel_width=0.3398058,
            rel_height=0.15267175,
        )
        self.main_menu_button.bind(on_press=self.main_menu)
        self.layout.add_widget(self.main_menu_button)

        self.buttons_for_board = []

    def main_menu(self, instance):
        self.manager.current = "main_menu"

    def display_sudoku(self, sudoku_data):
        """Update the Sudoku grid with new data"""
        self.buttons_for_board = []

        index = 0

        # This loop is kinda a mess, but it is just a way to make the board that is dynamic and work
        offset_y = 0  # This is the offset, that I need because the 4 pixels bold lines
        for row in range(9):
            offset_x = 0
            if row % 3 == 0 and row != 0:
                offset_y += 0.0024261844660194  # offset for the 4 pixels XD
            for column in range(9):
                if column % 3 == 0 and column != 0:
                    offset_x += 0.0043620501635768  # offset for the 4 pixels XD

                # Here I calculate in what 3x3 I am, I will need it later
                block_row = row // 3
                block_col = column // 3
                b = block_row * 3 + block_col

                data_from_photo = sudoku_data[index].data

                # This creates the button
                self.buttons_for_board.append(
                    Column(
                        name=f"{row},{column}",
                        rel_x=0.014563106
                        + offset_x
                        + 0.1067 * column,  # I need it like this
                        rel_y=0.69792802 - offset_y - 0.04798 * row,
                        rel_width=0.1067,
                        rel_height=0.0483,
                        block=f"{b}",
                    )
                )
                button = self.buttons_for_board[row * 9 + column]
                button.data = data_from_photo
                button.locked = True
                button.bind(
                    on_press=lambda instance, btn=button: self.pick_square(
                        instance, btn
                    )
                )

                self.layout.add_widget(self.buttons_for_board[row * 9 + column])
                index += 1

        for button in self.buttons_for_board:
            if button.data == 0:
                button.data = ""
            button.display_number(button.data)
            button.correct_font_size()

    def pick_square(self, instance, column):
        pass


# App class
class MainMenuApp(App):
    def build(self):
        Window.size = (412, 917)
        Window.fullscreen = False
        sm = ScreenManager(transition=NoTransition())
        game = GameScreen(name="game")
        editing_screen = EditingScreen(name="editing_screen")

        # Load and play background music
        sm.music = SoundLoader.load("background_music.mp3")
        if sm.music:
            sm.music.loop = True
            sm.music.play()
            sm.music.stop()

        # Main Screens
        sm.add_widget(MainMenuScreen(name="main_menu"))
        sm.add_widget(game)
        sm.add_widget(SolverScreen(name="solver"))

        # Screens for game
        sm.add_widget(ChoseDificultyScreen(name="chose_dificulty"))
        sm.add_widget(IncorrectSolutionScreen(name="incorrect_solution"))
        sm.add_widget(CorrectSolutionScreen(name="correct_solution"))

        # Screens for solver
        sm.add_widget(editing_screen)
        sm.add_widget(SudokuNotLoaded(name="sudoku_not_loaded"))
        sm.add_widget(SudokuNotSolvable(name="sudoku_not_solvable"))
        sm.add_widget(SudokuSolvedScreen(name="sudoku_solved"))

        Window.bind(on_resize=game.correct_font_size)
        Window.bind(on_resize=editing_screen.correct_font_size)
        if platform == "android":
            activity.bind(on_activity_result=self.solver_screen_intent_callback)
        return sm

    def solver_screen_intent_callback(self, request_code, result_code, intent):
        solver_screen = self.root.get_screen("solver")
        solver_screen.intent_callback(request_code, result_code, intent)


if __name__ == "__main__":
    MainMenuApp().run()
