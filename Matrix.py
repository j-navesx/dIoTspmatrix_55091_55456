from __future__ import annotations
from abc import ABC, abstractmethod
from Position import *
from typing import Union


class Matrix(ABC):

    @abstractmethod
    def __getitem__(self, item) -> float:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key, value):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __next__(self) -> Position:
        raise NotImplementedError

    @abstractmethod
    def __copy__(self) -> Matrix:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError

    def __add__(self, other) -> Union[float, int, Matrix]:
        if isinstance(other, (int, float)):
            return self._add_number(other)
        if isinstance(other, Matrix):
            return self._add_matrix(other)
        raise TypeError('__add__ invalid argument')

    @abstractmethod
    def _add_number(self, other: Union[int, float]) -> Matrix:
        raise NotImplementedError

    @abstractmethod
    def _add_matrix(self, val: Matrix) -> Matrix:
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self._mul_number(other)
        if isinstance(other, Matrix):
            return self._mul_matrix(other)
        raise ValueError('__mul__ invalid argument')

    @abstractmethod
    def _mul_number(self, other: Union[int, float]) -> Matrix:
        raise NotImplementedError

    @abstractmethod
    def _mul_matrix(self, other: Matrix) -> Matrix:
        raise NotImplementedError

    def __str__(self):
        dim = self.dim()
        string = ''
        if dim == (): return ''
        row_min, row_max = (dim[0][0],dim[0][1])
        col_min, col_max = (dim[1][0],dim[1][1])
        for row in range(row_min, row_max + 1 ):
            for col in range(col_min, col_max + 1):
                try:
                    string += "%.1f" % self[row,col]            
                except:
                    raise ValueError('__str__: invalid arguments')
                string += ' '
            string = string[:-1] + '\n'
        return string[:-1]


    @abstractmethod
    def dim(self) -> tuple[Position, ...]:
        raise NotImplementedError

    @abstractmethod
    def row(self, row: int) -> Matrix:
        raise NotImplementedError

    @abstractmethod
    def col(self, col: int) -> Matrix:
        raise NotImplementedError

    @abstractmethod
    def diagonal(self) -> Matrix:
        raise NotImplementedError

    @abstractmethod
    def transpose(self) -> Matrix:
        raise NotImplementedError
