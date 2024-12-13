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

        self.button_for_00 = button.Button([0, 0], 9, 231, 42, 42)
        self.button_for_10 = button.Button([1, 0], 52, 231, 44, 42)
        self.button_for_20 = button.Button([2, 0], 97, 231, 43, 42)
        self.button_for_30 = button.Button([3, 0], 144, 231, 42, 42)
        self.button_for_40 = button.Button([4, 0], 187, 231, 44, 42)
        self.button_for_50 = button.Button([5, 0], 232, 231, 43, 42)
        self.button_for_60 = button.Button([6, 0], 279, 231, 42, 42)
        self.button_for_70 = button.Button([7, 0], 322, 231, 44, 42)
        self.button_for_80 = button.Button([8, 0], 367, 231, 39, 42)

        self.button_for_01 = button.Button([0, 1], 9, 274, 42, 44)
        self.button_for_11 = button.Button([1, 1], 52, 274, 44, 44)
        self.button_for_21 = button.Button([2, 1], 97, 274, 43, 44)
        self.button_for_31 = button.Button([3, 1], 144, 274, 42, 44)
        self.button_for_41 = button.Button([4, 1], 187, 274, 44, 44)
        self.button_for_51 = button.Button([5, 1], 232, 274, 43, 44)
        self.button_for_61 = button.Button([6, 1], 279, 274, 42, 44)
        self.button_for_71 = button.Button([7, 1], 322, 274, 44, 44)
        self.button_for_81 = button.Button([8, 1], 367, 274, 39, 44)

        self.button_for_02 = button.Button([0, 2], 9, 319, 42, 43)
        self.button_for_12 = button.Button([1, 2], 52, 319, 44, 43)
        self.button_for_22 = button.Button([2, 2], 97, 319, 43, 43)
        self.button_for_32 = button.Button([3, 2], 144, 319, 42, 43)
        self.button_for_42 = button.Button([4, 2], 187, 319, 44, 43)
        self.button_for_52 = button.Button([5, 2], 232, 319, 43, 43)
        self.button_for_62 = button.Button([6, 2], 279, 319, 42, 43)
        self.button_for_72 = button.Button([7, 2], 322, 319, 44, 43)
        self.button_for_82 = button.Button([8, 2], 367, 319, 39, 43)

        self.button_for_03 = button.Button([0, 3], 9, 366, 42, 42)
        self.button_for_13 = button.Button([1, 3], 52, 366, 44, 42)
        self.button_for_23 = button.Button([2, 3], 97, 366, 43, 42)
        self.button_for_33 = button.Button([3, 3], 144, 366, 42, 42)
        self.button_for_43 = button.Button([4, 3], 187, 366, 44, 42)
        self.button_for_53 = button.Button([5, 3], 232, 366, 43, 42)
        self.button_for_63 = button.Button([6, 3], 279, 366, 42, 42)
        self.button_for_73 = button.Button([7, 3], 322, 366, 44, 42)
        self.button_for_83 = button.Button([8, 3], 367, 366, 39, 42)

        self.button_for_04 = button.Button([0, 4], 9, 409, 42, 44)
        self.button_for_14 = button.Button([1, 4], 52, 409, 44, 44)
        self.button_for_24 = button.Button([2, 4], 97, 409, 43, 44)
        self.button_for_34 = button.Button([3, 4], 144, 409, 42, 44)
        self.button_for_44 = button.Button([4, 4], 187, 409, 44, 44)
        self.button_for_54 = button.Button([5, 4], 232, 409, 43, 44)
        self.button_for_64 = button.Button([6, 4], 279, 409, 42, 44)
        self.button_for_74 = button.Button([7, 4], 322, 409, 44, 44)
        self.button_for_84 = button.Button([8, 4], 367, 409, 39, 44)

        self.button_for_05 = button.Button([0, 5], 9, 454, 42, 43)
        self.button_for_15 = button.Button([1, 5], 52, 454, 44, 43)
        self.button_for_25 = button.Button([2, 5], 97, 454, 43, 43)
        self.button_for_35 = button.Button([3, 5], 144, 454, 42, 43)
        self.button_for_45 = button.Button([4, 5], 187, 454, 44, 43)
        self.button_for_55 = button.Button([5, 5], 232, 454, 43, 43)
        self.button_for_65 = button.Button([6, 5], 279, 454, 42, 43)
        self.button_for_75 = button.Button([7, 5], 322, 454, 44, 43)
        self.button_for_85 = button.Button([8, 5], 367, 454, 39, 43)

        self.button_for_06 = button.Button([0, 6], 9, 501, 42, 42)
        self.button_for_16 = button.Button([1, 6], 52, 501, 44, 42)
        self.button_for_26 = button.Button([2, 6], 97, 501, 43, 42)
        self.button_for_36 = button.Button([3, 6], 144, 501, 42, 42)
        self.button_for_46 = button.Button([4, 6], 187, 501, 44, 42)
        self.button_for_56 = button.Button([5, 6], 232, 501, 43, 42)
        self.button_for_66 = button.Button([6, 6], 279, 501, 42, 42)
        self.button_for_76 = button.Button([7, 6], 322, 501, 44, 42)
        self.button_for_86 = button.Button([8, 6], 367, 501, 39, 42)

        self.button_for_07 = button.Button([0, 7], 9, 544, 42, 44)
        self.button_for_17 = button.Button([1, 7], 52, 544, 44, 44)
        self.button_for_27 = button.Button([2, 7], 97, 544, 43, 44)
        self.button_for_37 = button.Button([3, 7], 144, 544, 42, 44)
        self.button_for_47 = button.Button([4, 7], 187, 544, 44, 44)
        self.button_for_57 = button.Button([5, 7], 232, 544, 43, 44)
        self.button_for_67 = button.Button([6, 7], 279, 544, 42, 44)
        self.button_for_77 = button.Button([7, 7], 322, 544, 44, 44)
        self.button_for_87 = button.Button([8, 7], 367, 544, 39, 44)

        self.button_for_08 = button.Button([0, 8], 9, 589, 42, 39)
        self.button_for_18 = button.Button([1, 8], 52, 589, 44, 39)
        self.button_for_28 = button.Button([2, 8], 97, 589, 43, 39)
        self.button_for_38 = button.Button([3, 8], 144, 589, 42, 39)
        self.button_for_48 = button.Button([4, 8], 187, 589, 44, 39)
        self.button_for_58 = button.Button([5, 8], 232, 589, 43, 39)
        self.button_for_68 = button.Button([6, 8], 279, 589, 42, 39)
        self.button_for_78 = button.Button([7, 8], 322, 589, 44, 39)
        self.button_for_88 = button.Button([8, 8], 367, 589, 39, 39)

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

    def print_data(self):
        print(self.board_data)

    def draw(self, surface):
        temp_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        surface.blit(temp_surface, (self.x, self.y))
        return self.list_of_buttons
    