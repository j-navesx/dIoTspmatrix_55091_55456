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
        if not isinstance(pos, Position):
            raise TypeError("__getitem__: invalid arguments")
        return self._items.get(pos, self._zero)

    def __setitem__(self, pos: Union[Position, position], val: Union[int, float]):
        if not (isinstance(pos, Position) and isinstance(val, (int, float))):
            raise TypeError("__setitem__: invalid arguments")
        if(val == self._zero and self._items.get(pos) != None):
            self._items.pop(pos)
        if(val != self.zero):
            self._items.update({pos: val})

    def __len__(self) -> int:
        return len(self._items)

    def _add_number(self, other: Union[int, float]) -> Matrix:
        if not isinstance(other, (int, float)):
            raise TypeError("_add_number: invalid arguments")
        for key,item in self._items.items():
            self._items.update({key: item+other})
            
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
        positions = list(self)
        min_row = min(p[0] for p in positions)
        min_col = min(p[1] for p in positions)
        max_row = max(p[0] for p in positions)
        max_col = max(p[1] for p in positions)
        return ((min_row, min_col),(max_row,max_col))

    def row(self, row: int) -> Matrix:
        if not isinstance(row,int):
            raise ValueError('spmatrix_row: invalid arguments')

        row_matrix = MatrixSparseDOK(self._zero)
        
        col_max = self.max_row_col()[1]
        col_min = self.min_row_col()[1]

        for x in range(col_min,col_max + 1):
            if(self._items.get((row,x)) != None):
                self.__setitem__(Position(row,x),self._items.get((row,x)))
        return row_matrix

    def col(self, col: int) -> Matrix:
        if not isinstance(col,int):
            raise ValueError('spmatrix_col: invalid arguments')

        row_max = self.max_row_col()[0]
        row_min = self.min_row_col()[0]

        col_matrix = MatrixSparseDOK(self._zero)

        for x in range(row_min,row_max+1):
            if(self._items.get((x,col)) != None):
                self.__setitem__(Position(x,col),self._items.get((x,col)))
        return col_matrix

    def diagonal(self) -> Matrix:
        if not (self.square()):
            raise ValueError('spmatrix_diagonal: matrix not square')
        
        matrix = MatrixSparseDOK(self._zero)
    
        min_row, min_col = self.min_row_col()
        max_row, max_col = self.max_row_col()
        y = min_row
        for x in range(min_col,max_col+1):
            if(self._items.get((y,x)) != None):
                matrix.__setitem__(Position(y,x),self._items.get((y,x)))
            if(y < max_row):
                y += 1
        return matrix

    def max_row_col(self) -> tuple:
        first_iteration = True
        for key in self._items.keys():
            if(first_iteration):
                line = key[0]
                col = key[1]
                first_iteration = False
            if(key[0] > line):
                line = key[0]
            if(key[1] > col):
                col = key[1]
        return (line,col)

    def min_row_col(self) -> tuple:
        first_iteration = True
        for key in self._items.keys():
            if(first_iteration):
                line = key[0]
                col = key[1]
                first_iteration = False
            if(key[0] < line):
                line = key[0]
            if(key[1] < col):
                col = key[1]
        return (line,col)

    def square(self) -> bool:
        dim = self.dim()
        lines = dim[1][0] - dim[0][0] + 1
        col = dim[1][1] - dim[0][1] + 1
        print(lines)
        print(col)
        if(lines == col):
            return True
        else:
            return False

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

