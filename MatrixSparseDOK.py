from __future__ import annotations
from functools import reduce
from MatrixSparse import *
from Position import *
from typing import Union


class MatrixSparseDOK(MatrixSparse):
    def __init__(self, zero: float = 0):
        """Create a new sparse matrix with Dictionary of Keys (DOK)

        Args:
            zero (float): value to be used for zero elements
        """
        if not isinstance(zero, (float, int)):
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
            return False
        return self._items == other._items

    def __iter__(self):
        """Create an iterator for the matrix

        Returns:
            MatrixSparseDOK: an iterator for the matrix
        """
        self._iterator = iter(sorted(self._items))
        return self._iterator

    def __next__(self):
        """Get the next element of the matrix

        Returns:
            tuple[Position, float]: the next element of the matrix
        """
        return self._iterator.next()

    def __getitem__(self, pos: Union[Position, position]) -> float:
        """Get the value of the element at the given position

        Args:
            pos (Union[Position, position]): the position of the element

        Raises:
            ValueError: if the position is not valid

        Returns:
            float: the value of the element at the given position
        """
        if not isinstance(
            Position.convert_to_pos(Position, pos), Position
        ) and not isinstance(pos, Position):
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
        if not isinstance(
            Position.convert_to_pos(Position, pos), Position
        ) and not isinstance(pos, Position):
            raise ValueError("__setitem__() invalid arguments")
        if not isinstance(val, (int, float)):
            raise ValueError("__setitem__() invalid arguments")
        if not isinstance(pos, Position):
            pos = Position.convert_to_pos(Position, pos)
        if val == self._zero:
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
        mat._items = {key: value + other for key, value in self._items.items()}
        return mat

    def _add_matrix(self, other: MatrixSparseDOK) -> MatrixSparseDOK:
        """Add a matrix to the matrix

        Args:
            other (MatrixSparse): the matrix to add

        Returns:
            MatrixSparseDOK: the matrix with the other matrix added
        """
        if not isinstance(other, MatrixSparseDOK):
            raise ValueError("_add_matrix() invalid arguments")
        if self.zero != other.zero:
            raise ValueError("_add_matrix() incompatible matrices")
        dim1 = self.dim()
        dim2 = other.dim()
        size1_x = dim1[1][0] - dim1[0][0] + 1
        size1_y = dim1[1][1] - dim1[0][1] + 1
        size2_x = dim2[1][0] - dim2[0][0] + 1
        size2_y = dim2[1][1] - dim2[0][1] + 1
        if size1_x != size2_x or size1_y != size2_y:
            raise ValueError("_add_matrix() incompatible matrices")
        mat = MatrixSparseDOK(self.zero)
        mat._items = reduce(lambda d1, d2: {k: d1.get(k,0)+d2.get(k,0) for k in set(d1)|set(d2)}, [self._items, other._items])
        return mat

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
        return ((min_row, min_col), (max_row, max_col))

    def row(self, row: int) -> Matrix:
        """Get the row of the matrix

        Args:
            row (int): the row to get

        Raises:
            ValueError: if the row is not valid

        Returns:
            MatrixSparseDOK: the row of the matrix
        """
        if not isinstance(row, int):
            raise ValueError("spmatrix_row: invalid arguments")
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
        if not isinstance(col, int):
            raise ValueError("spmatrix_col: invalid arguments")
        mat = MatrixSparseDOK(self._zero)
        mat._items = {key: value for key, value in self._items.items() if key[1] == col}
        return mat

    def diagonal(self) -> Matrix:
        """Get the diagonal of the matrix

        Returns:
            MatrixSparseDOK: the diagonal of the matrix
        """
        if not (self.square()):
            raise ValueError("spmatrix_diagonal: matrix not square")
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
        if not isinstance(size, int) or size < 1:
            raise ValueError("eye() invalid parameters")
        if not isinstance(unitary, (int, float)) or not isinstance(zero, (int, float)):
            raise ValueError("eye() invalid parameters")
        mat = MatrixSparseDOK(zero)
        for i in range(size):
            mat[i, i] = unitary
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
        if self.sparsity() < 0.5:
            raise ValueError("compress() dense matrix")
        dim = self.dim()
        upper_left_pos = dim[0]
        zero = self._zero
        offsets = []
        # Auxiliary variables
        rows = []
        indexs = []
        # Getting the rows and indexs of the matrix into auxiliary variables
        for row in range(dim[0][0], dim[1][0]+1):
            r = self.row(row)
            li = []
            ind = []
            for col in range(dim[0][1], dim[1][1]+1):
                li.append(r[row,col])
                ind.append(row)
            rows.append(li)
            indexs.append(ind)
        
        counts = [x.count(zero) for x in rows]
        # Order the rows based on density
        rows = sorted(rows, key=lambda x: x.count(zero))
        # Assign first denser row
        merged = rows[0]
        # Order the indexes based on density
        indexs = [i for _,i in sorted(zip(counts,indexs))]
        # Assign first denser index
        index_list = indexs[0]
        # Calculate the first offset and add it
        for val in merged:
            if val != zero:
                offsets.append(merged.index(val))
                break
        # Merge the rest of the rows
        for i in range(1,len(rows)):
            if not all(x == zero for x in rows[i]):
                offset, merged, index_list = self.compress_merge_offset(0, merged, rows[i], index_list, indexs[i], zero)
            else:
                offset = 0
            if offset == len(merged):
                merged += rows[i]
                index_list += indexs[i]
            offsets.append(offset)
        values = merged
        indexes = index_list
        # Mark uneccessary zeros
        for val in reversed(list(merged)):
            if val != zero:
                break
            indexes[merged.index(val)] = -1
        # Order offsets
        order_of_indexes = [ind[0] for ind in indexs]
        offsets = [i for _,i in sorted(zip(order_of_indexes,offsets))]
        return (upper_left_pos, zero, tuple(values), tuple(indexes), tuple(offsets))

    def compress_merge_offset(self,offset, list1, list2, row1, row2, zero):
        a = [1 if v!=zero else 0 for v in list1]
        a += [0 for _ in list2]
        b = [0 for _ in range(offset)]
        b += [2 if v!=zero else 0 for v in list2]
        c = [sum(x) for x in zip(a,b)]
        if 3 in c:
            return self.compress_merge_offset(offset+1, list1, list2, row1, row2, zero)
        else:
            merged = []
            indexes = []
            list2 = [zero for _ in range(offset)] + list2
            row2 = [row2[0] for _ in range(len(list2))]
            for i in range(len(list2)):
                if i < len(list1):
                    if list1[i] == zero:
                        merged.append(list2[i])
                        indexes.append(row2[0])
                    else:
                        merged.append(list1[i])
                        indexes.append(row1[i])
                else:
                    merged.append(list2[i])
                    indexes.append(row2[i])

            return offset, merged, indexes

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
