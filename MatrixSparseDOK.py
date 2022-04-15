from __future__ import annotations
from MatrixSparse import *
from Position import *
from typing import Union



class MatrixSparseDOK(MatrixSparse):

    def __init__(self, zero: float = 0.0):
        if not isinstance(zero, (float,int)):
            raise TypeError("__init__: invalid arguments")
        self._zero = zero
        self._items = dict()
    
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
        # TODO: implement this method
        pass

    def __setitem__(self, pos: Union[Position, position], val: Union[int, float]):
        # TODO: implement this method
        pass

    def __len__(self) -> int:
        # TODO: implement this method
        pass

    def _add_number(self, other: Union[int, float]) -> Matrix:
        # TODO: implement this method
        pass

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
        # TODO: implement this method
        pass

    def row(self, row: int) -> Matrix:
        # TODO: implement this method
        pass

    def col(self, col: int) -> Matrix:
        # TODO: implement this method
        pass

    def diagonal(self) -> Matrix:
        # TODO: implement this method
        pass

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

