from __future__ import annotations
from multiprocessing.sharedctypes import Value
from MatrixSparse import *
from Position import *
from typing import Union


class MatrixSparseDOK(MatrixSparse):

    def __init__(self, zero: float = 0):
        if not isinstance(zero, (float,int)):
            raise ValueError("__init__() invalid arguments")
        super(MatrixSparseDOK, self).__init__(zero)
        self._items = {}

    @MatrixSparse.zero.setter
    def zero(self, val: Union[int, float]):
        MatrixSparse.zero.fset(self, val)
        self._items = {key: value for key, value in self._items.items() if (value != self._zero)}

    def __copy__(self):
        copy = MatrixSparseDOK(self.zero)
        copy._items = self._items
        return copy

    def __eq__(self, other: MatrixSparseDOK):
        if not isinstance(other, MatrixSparseDOK):
            raise TypeError("__eq__: invalid arguments")
        return self == other

    def __iter__(self):
        self.iterator = iter(self._items)
        return self.iterator

    def __next__(self):
        return self.iterator.next()

    def __getitem__(self, pos: Union[Position, position]) -> float:
        if not isinstance(Position.convert_to_pos(Position,pos), Position) and not isinstance(pos, Position):
            raise ValueError("__getitem__() invalid arguments")
        if not isinstance(pos, Position):
            pos = Position.convert_to_pos(Position, pos)
        return self._items.get(pos, self._zero)

    def __setitem__(self, pos: Union[Position, position], val: Union[int, float]):
        if not isinstance(Position.convert_to_pos(Position, pos), Position) and not isinstance(pos, Position):
            raise ValueError("__setitem__() invalid arguments")
        if not isinstance(val, (int, float)):
            raise ValueError("__setitem__() invalid arguments")
        if not isinstance(pos, Position):
            pos = Position.convert_to_pos(Position, pos)
        if(val == self._zero):
            self._items.pop(pos, 1)
        else:
            self._items.update({pos: val})

    def __len__(self) -> int:
        return len(self._items)

    def _add_number(self, other: Union[int, float]) -> Matrix:
        if not isinstance(other, (int, float)):
            raise ValueError("_add_number: invalid arguments")
        mat = MatrixSparseDOK(self._zero)
        mat._items = {key: value+other for key, value in self._items.items()}
        return mat
            
    def _add_matrix(self, other: MatrixSparse) -> MatrixSparse:
        # TODO: implement this method
        pass

    def _mul_number(self, other: Union[int, float]) -> Matrix:
        # TODO: implement this method
        pass

    def _mul_matrix(self, other: MatrixSparse) -> MatrixSparse:
        # TODO: implement this method
        pass

    def dim(self) -> tuple[Position, ...]:
        """Compute the dimensions of the matrix

        Returns:
            tuple[Position, ...]: The dimensions of the matrix as a tuple of Position objects
        """
        if len(self) == 0:
            return tuple()
        positions = list(self._items)
        min_row = min(p[0] for p in positions)
        min_col = min(p[1] for p in positions)
        max_row = max(p[0] for p in positions)
        max_col = max(p[1] for p in positions)
        return ((min_row, min_col),(max_row,max_col))

    def row(self, row: int) -> Matrix:
        if not isinstance(row,int):
            raise ValueError('spmatrix_row: invalid arguments')
        mat = MatrixSparseDOK(self._zero)
        mat._items = {key: value for key, value in self._items.items() if key[0] == row}
        return mat

    def col(self, col: int) -> Matrix:
        if not isinstance(col,int):
            raise ValueError('spmatrix_col: invalid arguments')
        mat = MatrixSparseDOK(self._zero)
        mat._items = {key: value for key, value in self._items.items() if key[1] == col}
        return mat

    def diagonal(self) -> Matrix:
        if not (self.square()):
            raise ValueError('spmatrix_diagonal: matrix not square')
        mat = MatrixSparseDOK(self._zero)
        mat._items = {key: value for key, value in self._items.items() if key[0] == key[1]}
        return mat

    def square(self) -> bool:
        dim = self.dim()
        lines = dim[1][0] - dim[0][0] + 1
        col = dim[1][1] - dim[0][1] + 1
        return lines == col

    @staticmethod
    def eye(size: int, unitary: float = 1.0, zero: float = 0.0) -> MatrixSparseDOK:
        # TODO: implement this method
        pass

    def transpose(self) -> MatrixSparseDOK:
        # TODO: implement this method
        pass

    def compress(self) -> compressed:
        # TODO: implement this method
        pass

    @staticmethod
    def doi(compressed_vector: compressed, pos: Position) -> float:
        # TODO: implement this method
        pass

    @staticmethod
    def decompress(compressed_vector: compressed) -> MatrixSparse:
        # TODO: implement this method
        pass

