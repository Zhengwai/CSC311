"""CSC108: Fall 2021 -- Assignment 1: Unscramble

This code is provided solely for the personal and private use of students
taking the CSC108 course at the University of Toronto. Copying for purposes
other than this use is expressly prohibited. All forms of distribution of
this code, whether as given or with any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2021 Michelle Craig, Tom Fairgrieve, Sadia Sharmin, and
Jacqueline Smith.
"""

# Move constants
SHIFT = 'S'
FLIP = 'F'
CHECK = 'C'

# Constant for hint functions
HINT_MODE_SECTION_LENGTH = 3


def get_section_start(section_num: int, section_len: int) -> int:
    """Return the starting index of the section corresponding to section_num
    if the length of a section is section_len.

    >>> get_section_start(1, 3)
    0
    >>> get_section_start(2, 3)
    3
    >>> get_section_start(3, 3)
    6
    >>> get_section_start(4, 3)
    9
    """

    return (section_num - 1) * section_len


def is_valid_move(move: str) -> bool:
    """Return True if and only if move is a valid move.

    >>> is_valid_move('S')
    True
    >>> is_valid_move('F')
    True
    >>> is_valid_move('C')
    True
    """

    return move in (SHIFT, FLIP, CHECK)


def get_num_sections(answer: str, section_len: int) -> int:
    """Return the number of sections in the answer string based on the section
    length.

    >>> get_num_sections('BANANA', 3)
    2
    >>> get_num_sections('APPLEWATCH', 2)
    5
    >>> get_num_sections('PEOPLE', 2)
    3
    """

    return len(answer) // section_len


def is_valid_section(section_num: int, answer: str, section_len: int) -> bool:
    """Return True if and only if the section number represents a section
    number that is valid for the given answer and section length.

    >>> is_valid_section(2, 'BANANA', 3)
    True
    >>> is_valid_section(5, 'PEOPLE', 2)
    False
    >>> is_valid_section(5, 'APPLEWATCH', 2)
    True
    """
    if 1 <= section_num <= get_num_sections(answer, section_len):
        return True
    else:
        return False


def check_section(state: str, answer: str, section_num: int,
                  section_len: int) -> bool:
    """Return True if and only if the specified section in the game
    has been correctly unscrambled.


    >>> check_section('BANNAA', 'BANANA', 1, 3)
    True
    >>> check_section('PEOPLE', 'PEOPLE', 2, 3)
    True
    >>> check_section('APPLEWTACH', 'APPLEWATCH', 2, 5)
    False
    """

    index = get_section_start(section_num, section_len)
    if state[index: index + section_len] == answer[index: index + section_len]:
        return True
    else:
        return False


def change_section(state: str, move: str, section_num: int,
                   section_len: int) -> str:
    """Return a updated game state after applying a FLIP or SHIFT move
    to the specified section.

    >>> change_section('BANAAN', 'S', 2, 3)
    'BANANA'
    >>> change_section('OEPPLE', 'F', 1, 3)
    'PEOPLE'
    >>> change_section('APPLEHWATC', 'S', 2, 5)
    'APPLEWATCH'
    """

    if move == FLIP:
        index = get_section_start(section_num, section_len)
        first = state[index]
        last = state[index + section_len - 1]
        mid = state[index + 1: index + section_len - 1]
        word = last + mid + first
        return state[:index] + word + state[index + section_len:]
    if move == SHIFT:
        index = get_section_start(section_num, section_len)
        first = state[index]
        rest = state[index + 1: index + section_len]
        word = rest + first
        return state[:index] + word + state[index + section_len:]
    return None


def section_needs_flip(state: str, answer: str, section_num: int) -> bool:
    """Return True if and only if the specified section in the game state will
    never match the same section in the answer string without doing a FLIP move.

    >>> section_needs_flip('OEPPLE', 'PEOPLE', 1)
    True
    >>> section_needs_flip('BNAANA', 'BANANA', 1)
    True
    >>> section_needs_flip('VFAOUR', 'FAVOUR', 1)
    False
    """

    index = get_section_start(section_num, HINT_MODE_SECTION_LENGTH)

    s_1 = change_section(state, "S", section_num, HINT_MODE_SECTION_LENGTH)
    s_2 = change_section(s_1, "S", section_num, HINT_MODE_SECTION_LENGTH)
    answer_sec = answer[index: index + HINT_MODE_SECTION_LENGTH]
    if answer_sec not in (state[index: index + HINT_MODE_SECTION_LENGTH],
                          s_1[index: index + HINT_MODE_SECTION_LENGTH],
                          s_2[index: index + HINT_MODE_SECTION_LENGTH]):
        return True
    return False


def get_move_hint(state: str, answer: str, section_num: int) -> str:
    """Return a move that will help the player rearrange the specified section
    correctly.

    >>> get_move_hint('OEPPLE', 'PEOPLE', 1)
    'F'
    >>> get_move_hint('BNAANA', 'BANANA', 1)
    'F'
    >>> get_move_hint('VFAOUR', 'FAVOUR', 1)
    'S'
    """

    if section_needs_flip(state, answer, section_num):
        return 'F'
    return 'S'
