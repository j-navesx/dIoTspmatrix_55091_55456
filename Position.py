from __future__ import annotations


class Position:
    _pos = tuple[int, int]

    def __init__(self, row: int, col: int):
        """Create a new position

        Args:
            row (int): row number
            col (int): column number

        Raises:
            TypeError: invalid type of arguments
            ValueError: invalid value for arguments
        """
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError("__init__() invalid arguments")
        if row < 0 or col < 0:
            raise ValueError("__init__() invalid arguments")
        self._pos = (row, col)

    def __str__(self):
        """Defines how to print the position

        Returns:
            str: position in string format
        """
        return str(self._pos)

    def __repr__(self):
        """Defines representation of the position

        Returns:
            str: position in string format
        """
        return str(self._pos)

    def __hash__(self):
        """Defines hash of the position

        Returns:
            int: hash of the position
        """
        return hash(self._pos)

    def __getitem__(self, item: int) -> int:
        """Defines how to get the value of the position
        
        Args:
            item (int): index of the position

        Returns:
            int: value of the position
        """
        if isinstance(item, int) and (0 <= item <= 1):
            return self._pos[item]
        raise IndexError("__getitem__() invalid arguments")

    def __eq__(self, other: Position):
        """Defines how to compare the position

        Args:
            other (Position): position to compare

        Returns:
            bool: True if the position is equal, False otherwise
        """
        if isinstance(other, Position):
            return self[0] == other[0] and self[1] == other[1]
        raise TypeError("__eq__() invalid arguments")
