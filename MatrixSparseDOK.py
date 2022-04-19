from __future__ import annotations
from multiprocessing.sharedctypes import Value
from MatrixSparse import *
from Position import *
from typing import Union


class MatrixSparseDOK(MatrixSparse):

    def __init__(self, zero: float = 0):
        """Create a new sparse matrix with Dictionary of Keys (DOK)

        Args:
            zero (float): value to be used for zero elements
        """
        if not isinstance(zero, (float,int)):
            raise ValueError("__init__() invalid arguments")
        super(MatrixSparseDOK, self).__init__(zero)
        self._items = {}

    @MatrixSparse.zero.setter
    def zero(self, val: Union[int, float]):
        """Set the value to be used for zero elements

        Args:
            val (Union[int, float]): value to be used for zero elements
        """
        MatrixSparse.zero.fset(self, val)
        self._items = {key: value for key, value in self._items.items() if (value != self._zero)}

    def __copy__(self):
        """Create a copy of the matrix

        Returns:
            MatrixSparseDOK: a copy of the matrix
        """
        copy = MatrixSparseDOK(self.zero)
        copy._items = self._items
        return copy

    def __eq__(self, other: MatrixSparseDOK):
        """Compare two sparse matrices

        Args:
            other (MatrixSparseDOK): the matrix to compare with

        Raises:
            ValueError: if the other matrix is not a sparse matrix

        Returns:
            bool: True if the two matrices are equal, False otherwise
        """
        if not isinstance(other, MatrixSparseDOK):
            raise ValueError("__eq__: invalid arguments")
        return self._items == other._items

    def __iter__(self):
        """Create an iterator for the matrix

        Returns:
            MatrixSparseDOK: an iterator for the matrix
        """
        self.iterator = iter(sorted(self._items))
        return self.iterator

    def __next__(self):
        """Get the next element of the matrix

        Returns:
            tuple[Position, float]: the next element of the matrix
        """
        return self.iterator.next()

    def __getitem__(self, pos: Union[Position, position]) -> float:
        """Get the value of the element at the given position

        Args:
            pos (Union[Position, position]): the position of the element

        Raises:
            ValueError: if the position is not valid

        Returns:
            float: the value of the element at the given position
        """
        if not isinstance(Position.convert_to_pos(Position,pos), Position) and not isinstance(pos, Position):
            raise ValueError("__getitem__() invalid arguments")
        if not isinstance(pos, Position):
            pos = Position.convert_to_pos(Position, pos)
        return self._items.get(pos, self._zero)

    def __setitem__(self, pos: Union[Position, position], val: Union[int, float]):
        """Set the value of the element at the given position

        Args:
            pos (Union[Position, position]): the position of the element
            val (Union[int, float]): the value of the element
        
        Raises:
            ValueError: if the position is not valid
        
        Returns:
            None: if the value is zero
        """
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
        """Get the number of elements in the matrix

        Returns:
            int: the number of elements in the matrix
        """
        return len(self._items)

    def _add_number(self, other: Union[int, float]) -> Matrix:
        """Add a number to the matrix

        Args:
            other (Union[int, float]): the number to add

        Raises:
            ValueError: if the other number is not a number

        Returns:
            MatrixSparseDOK: the matrix with the number added
        """
        if not isinstance(other, (int, float)):
            raise ValueError("_add_number: invalid arguments")
        mat = MatrixSparseDOK(self._zero)
        mat._items = {key: value+other for key, value in self._items.items()}
        return mat
            
    def _add_matrix(self, other: MatrixSparse) -> MatrixSparse:
        """Add a matrix to the matrix

        Args:
            other (MatrixSparse): the matrix to add

        Returns:
            MatrixSparseDOK: the matrix with the other matrix added
        """
        # TODO: implement this method
        pass

    def _mul_number(self, other: Union[int, float]) -> Matrix:
        """Multiply the matrix by a number

        Args:
            other (Union[int, float]): the number to multiply by
        
        Raises:
            ValueError: if the other number is not a number
        
        Returns:
            MatrixSparseDOK: the matrix with the number multiplied
        """
        # TODO: implement this method
        pass

    def _mul_matrix(self, other: MatrixSparse) -> MatrixSparse:
        """Multiply the matrix by another matrix

        Args:
            other (MatrixSparse): the matrix to multiply by

        Returns:
            MatrixSparseDOK: the matrix with the other matrix multiplied
        """
        # TODO: implement this method
        pass

    def dim(self) -> tuple[Position, ...]:
        """Get the dimensions of the matrix
        
        Returns:
            tuple[Position, ...]: the dimensions of the matrix
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
        """Get the row of the matrix

        Args:
            row (int): the row to get

        Raises:
            ValueError: if the row is not valid

        Returns:
            MatrixSparseDOK: the row of the matrix
        """
        if not isinstance(row,int):
            raise ValueError('spmatrix_row: invalid arguments')
        mat = MatrixSparseDOK(self._zero)
        mat._items = {key: value for key, value in self._items.items() if key[0] == row}
        return mat

    def col(self, col: int) -> Matrix:
        """Get the column of the matrix
        
        Args:
            col (int): the column to get
            
        Raises:
            ValueError: if the column is not valid

        Returns:
            MatrixSparseDOK: the column of the matrix
        """
        if not isinstance(col,int):
            raise ValueError('spmatrix_col: invalid arguments')
        mat = MatrixSparseDOK(self._zero)
        mat._items = {key: value for key, value in self._items.items() if key[1] == col}
        return mat

    def diagonal(self) -> Matrix:
        """Get the diagonal of the matrix

        Returns:
            MatrixSparseDOK: the diagonal of the matrix
        """
        if not (self.square()):
            raise ValueError('spmatrix_diagonal: matrix not square')
        mat = MatrixSparseDOK(self._zero)
        mat._items = {key: value for key, value in self._items.items() if key[0] == key[1]}
        return mat

    def square(self) -> bool:
        """Check if the matrix is square

        Returns:
            bool: True if the matrix is square, False otherwise
        """
        dim = self.dim()
        lines = dim[1][0] - dim[0][0] + 1
        col = dim[1][1] - dim[0][1] + 1
        return lines == col

    @staticmethod
    def eye(size: int, unitary: float = 1.0, zero: float = 0.0) -> MatrixSparseDOK:
        """Create an identity matrix
            
        Args:
            size (int): the size of the matrix
            unitary (float): the value of the unitary elements
            zero (float): the value of the zero elements

        Returns:
            MatrixSparseDOK: the identity matrix
        """
        mat = MatrixSparseDOK(zero)
        for i in range(size):
            mat[i,i] = unitary
        return mat

    def transpose(self) -> MatrixSparseDOK:
        """Transpose the matrix
            
            Returns:
                MatrixSparseDOK: the transpose of the matrix
            """
        mat = MatrixSparseDOK(self._zero)
        for key, value in self._items.items():
            mat[key[1], key[0]] = value
        return mat

    def compress(self) -> compressed:
        """Compress the matrix

        Returns:
            compressed: the compressed matrix
        """
        # TODO: implement this method
        pass

    @staticmethod
    def doi(compressed_vector: compressed, pos: Position) -> float:
        """Get the value of the element at the given position
            
        Args:
            compressed_vector (compressed): the compressed vector
            pos (Position): the position of the element

        Returns:
            float: the value of the element at the given position
        """
        # TODO: implement this method
        pass

    @staticmethod
    def decompress(compressed_vector: compressed) -> MatrixSparse:
        """Decompress the matrix

        Args:
            compressed_vector (compressed): the compressed matrix

        Returns:
            MatrixSparseDOK: the decompressed matrix
        """
        # TODO: implement this method
        pass

