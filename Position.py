from __future__ import annotations


class Position:
    _pos = tuple[int, int]

    def __init__(self, row: int, col: int):
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError("__init__: invalid arguments")
        if row < 0 or col < 0:
            raise ValueError("__init__: invalid arguments")
        self._pos = (row, col)

    def __str__(self):
        return str(self._pos)

    def __repr__(self):
        return str(self._pos)

    def __hash__(self):
        return hash(self._pos)

    def __getitem__(self, item: int) -> int:
        if isinstance(item, int) and (0 <= item <= 1):
            return self._pos[item]
        raise IndexError("__getitem__: invalid arguments")

    def __eq__(self, other: Position):
        if isinstance(other, Position):
            return self[0] == other[0] and self[1] == other[1]
        raise TypeError("__eq__: invalid arguments")
