import pygame
import button


class Board:
    """Class for the board."""

    def __init__(self, board_state):
        self.height = 400
        self.width = 400
        self.board_state = board_state

        self.x = 2
        self.y = 220

        self.board_surface = pygame.Surface((self.width, self.height))
        self.board_data = [[" " for _ in range(9)] for _ in range(9)]

        self.button_for_00 = button.Button([0, 0], 6, 223, 44, 44)
        self.button_for_10 = button.Button([1, 0], 50, 223, 44, 44)
        self.button_for_20 = button.Button([2, 0], 94, 223, 44, 44)
        self.button_for_30 = button.Button([3, 0], 140, 223, 44, 44)
        self.button_for_40 = button.Button([4, 0], 184, 223, 44, 44)
        self.button_for_50 = button.Button([5, 0], 228, 223, 44, 44)
        self.button_for_60 = button.Button([6, 0], 274, 223, 44, 44)
        self.button_for_70 = button.Button([7, 0], 318, 223, 44, 44)
        self.button_for_80 = button.Button([8, 0], 362, 223, 44, 44)

        self.button_for_01 = button.Button([0, 1], 6, 267, 44, 44)
        self.button_for_11 = button.Button([1, 1], 50, 267, 44, 44)
        self.button_for_21 = button.Button([2, 1], 94, 267, 44, 44)
        self.button_for_31 = button.Button([3, 1], 140, 267, 44, 44)
        self.button_for_41 = button.Button([4, 1], 184, 267, 44, 44)
        self.button_for_51 = button.Button([5, 1], 228, 267, 44, 44)
        self.button_for_61 = button.Button([6, 1], 274, 267, 44, 44)
        self.button_for_71 = button.Button([7, 1], 318, 267, 44, 44)
        self.button_for_81 = button.Button([8, 1], 362, 267, 44, 44)

        self.button_for_02 = button.Button([0, 2], 6, 311, 44, 44)
        self.button_for_12 = button.Button([1, 2], 50, 311, 44, 44)
        self.button_for_22 = button.Button([2, 2], 94, 311, 44, 44)
        self.button_for_32 = button.Button([3, 2], 140, 311, 44, 44)
        self.button_for_42 = button.Button([4, 2], 184, 311, 44, 44)
        self.button_for_52 = button.Button([5, 2], 228, 311, 44, 44)
        self.button_for_62 = button.Button([6, 2], 274, 311, 44, 44)
        self.button_for_72 = button.Button([7, 2], 318, 311, 44, 44)
        self.button_for_82 = button.Button([8, 2], 362, 311, 44, 44)

        self.button_for_03 = button.Button([0, 3], 6, 357, 44, 44)
        self.button_for_13 = button.Button([1, 3], 50, 357, 44, 44)
        self.button_for_23 = button.Button([2, 3], 94, 357, 44, 44)
        self.button_for_33 = button.Button([3, 3], 140, 357, 44, 44)
        self.button_for_43 = button.Button([4, 3], 184, 357, 44, 44)
        self.button_for_53 = button.Button([5, 3], 228, 357, 44, 44)
        self.button_for_63 = button.Button([6, 3], 274, 357, 44, 44)
        self.button_for_73 = button.Button([7, 3], 318, 357, 44, 44)
        self.button_for_83 = button.Button([8, 3], 362, 357, 44, 44)

        self.button_for_04 = button.Button([0, 4], 6, 401, 44, 44)
        self.button_for_14 = button.Button([1, 4], 50, 401, 44, 44)
        self.button_for_24 = button.Button([2, 4], 94, 401, 44, 44)
        self.button_for_34 = button.Button([3, 4], 140, 401, 44, 44)
        self.button_for_44 = button.Button([4, 4], 184, 401, 44, 44)
        self.button_for_54 = button.Button([5, 4], 228, 401, 44, 44)
        self.button_for_64 = button.Button([6, 4], 274, 401, 44, 44)
        self.button_for_74 = button.Button([7, 4], 318, 401, 44, 44)
        self.button_for_84 = button.Button([8, 4], 362, 401, 44, 44)

        self.button_for_05 = button.Button([0, 5], 6, 445, 44, 44)
        self.button_for_15 = button.Button([1, 5], 50, 445, 44, 44)
        self.button_for_25 = button.Button([2, 5], 94, 445, 44, 44)
        self.button_for_35 = button.Button([3, 5], 140, 445, 44, 44)
        self.button_for_45 = button.Button([4, 5], 184, 445, 44, 44)
        self.button_for_55 = button.Button([5, 5], 228, 445, 44, 44)
        self.button_for_65 = button.Button([6, 5], 274, 445, 44, 44)
        self.button_for_75 = button.Button([7, 5], 318, 445, 44, 44)
        self.button_for_85 = button.Button([8, 5], 362, 445, 44, 44)

        self.button_for_06 = button.Button([0, 6], 6, 491, 44, 44)
        self.button_for_16 = button.Button([1, 6], 50, 491, 44, 44)
        self.button_for_26 = button.Button([2, 6], 94, 491, 44, 44)
        self.button_for_36 = button.Button([3, 6], 140, 491, 44, 44)
        self.button_for_46 = button.Button([4, 6], 184, 491, 44, 44)
        self.button_for_56 = button.Button([5, 6], 228, 491, 44, 44)
        self.button_for_66 = button.Button([6, 6], 274, 491, 44, 44)
        self.button_for_76 = button.Button([7, 6], 318, 491, 44, 44)
        self.button_for_86 = button.Button([8, 6], 362, 491, 44, 44)

        self.button_for_07 = button.Button([0, 7], 6, 535, 44, 44)
        self.button_for_17 = button.Button([1, 7], 50, 535, 44, 44)
        self.button_for_27 = button.Button([2, 7], 94, 535, 44, 44)
        self.button_for_37 = button.Button([3, 7], 140, 535, 44, 44)
        self.button_for_47 = button.Button([4, 7], 184, 535, 44, 44)
        self.button_for_57 = button.Button([5, 7], 228, 535, 44, 44)
        self.button_for_67 = button.Button([6, 7], 274, 535, 44, 44)
        self.button_for_77 = button.Button([7, 7], 318, 535, 44, 44)
        self.button_for_87 = button.Button([8, 7], 362, 535, 44, 44)

        self.button_for_08 = button.Button([0, 8], 6, 579, 44, 44)
        self.button_for_18 = button.Button([1, 8], 50, 579, 44, 44)
        self.button_for_28 = button.Button([2, 8], 94, 579, 44, 44)
        self.button_for_38 = button.Button([3, 8], 140, 579, 44, 44)
        self.button_for_48 = button.Button([4, 8], 184, 579, 44, 44)
        self.button_for_58 = button.Button([5, 8], 228, 579, 44, 44)
        self.button_for_68 = button.Button([6, 8], 274, 579, 44, 44)
        self.button_for_78 = button.Button([7, 8], 318, 579, 44, 44)
        self.button_for_88 = button.Button([8, 8], 362, 579, 44, 44)

        self.list_of_buttons = [
            self.button_for_00,
            self.button_for_10,
            self.button_for_20,
            self.button_for_30,
            self.button_for_40,
            self.button_for_50,
            self.button_for_60,
            self.button_for_70,
            self.button_for_80,
            self.button_for_01,
            self.button_for_11,
            self.button_for_21,
            self.button_for_31,
            self.button_for_41,
            self.button_for_51,
            self.button_for_61,
            self.button_for_71,
            self.button_for_81,
            self.button_for_02,
            self.button_for_12,
            self.button_for_22,
            self.button_for_32,
            self.button_for_42,
            self.button_for_52,
            self.button_for_62,
            self.button_for_72,
            self.button_for_82,
            self.button_for_03,
            self.button_for_13,
            self.button_for_23,
            self.button_for_33,
            self.button_for_43,
            self.button_for_53,
            self.button_for_63,
            self.button_for_73,
            self.button_for_83,
            self.button_for_04,
            self.button_for_14,
            self.button_for_24,
            self.button_for_34,
            self.button_for_44,
            self.button_for_54,
            self.button_for_64,
            self.button_for_74,
            self.button_for_84,
            self.button_for_05,
            self.button_for_15,
            self.button_for_25,
            self.button_for_35,
            self.button_for_45,
            self.button_for_55,
            self.button_for_65,
            self.button_for_75,
            self.button_for_85,
            self.button_for_06,
            self.button_for_16,
            self.button_for_26,
            self.button_for_36,
            self.button_for_46,
            self.button_for_56,
            self.button_for_66,
            self.button_for_76,
            self.button_for_86,
            self.button_for_07,
            self.button_for_17,
            self.button_for_27,
            self.button_for_37,
            self.button_for_47,
            self.button_for_57,
            self.button_for_67,
            self.button_for_77,
            self.button_for_87,
            self.button_for_08,
            self.button_for_18,
            self.button_for_28,
            self.button_for_38,
            self.button_for_48,
            self.button_for_58,
            self.button_for_68,
            self.button_for_78,
            self.button_for_88,
        ]

    def append_to_board(self, y, x, value):
        self.board_data[y][x] = value

    def draw(self, surface):
        temp_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        surface.blit(temp_surface, (self.x, self.y))
        return self.list_of_buttons

    def draw_number_on_button_placing_numbers(
        self, button_instance, surface_instance, number
    ):
        button_surface = pygame.Surface(
            (button_instance.rect.width, button_instance.rect.height), pygame.SRCALPHA
        )
        image = pygame.image.load(f"textures/{number}.png")
        image = image.convert_alpha()

        image_rect = image.get_rect(center=button_surface.get_rect().center)

        button_surface.blit(image, image_rect.topleft)
        surface_instance.blit(button_surface, button_instance.rect.topleft)

    def draw_number_on_button_notes(self, button_instance, surface_instance, number):
        button_surface = pygame.Surface(
            (button_instance.rect.width, button_instance.rect.height), pygame.SRCALPHA
        )
        image = pygame.image.load(f"textures/small_{number}.png")
        image = image.convert_alpha()

        image_rect = image.get_rect()

        if number == 1:
            image_rect.topleft = (3, 3)  # Top-left
        elif number == 2:
            image_rect.topleft = (19, 3)  # Top-center
        elif number == 3:
            image_rect.topleft = (35, 3)  # Top-right
        elif number == 4:
            image_rect.topleft = (2, 17)  # Middle-left
        elif number == 5:
            image_rect.topleft = (19, 17)  # Center
        elif number == 6:
            image_rect.topleft = (35, 17)  # Middle-right
        elif number == 7:
            image_rect.topleft = (3, 31)  # Bottom-left
        elif number == 8:
            image_rect.topleft = (19, 31)  # Bottom-center
        elif number == 9:
            image_rect.topleft = (
                35,
                31,
            )

        button_surface.blit(image, image_rect.topleft)
        surface_instance.blit(button_surface, button_instance.rect.topleft)
