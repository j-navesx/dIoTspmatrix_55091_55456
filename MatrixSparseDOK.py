from __future__ import annotations
from functools import reduce
from numpy import trunc
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
        if not isinstance(val, (int, float)):
            raise ValueError("__setitem__() invalid arguments")
        if not isinstance(pos, Position):
            if not isinstance(Position.convert_to_pos(Position, pos), Position
        ) and not isinstance(pos, Position):
                raise ValueError("__setitem__() invalid arguments")
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
        if not isinstance(other, (int, float)):
            raise ValueError("_mul_number: invalid arguments")
        mat = MatrixSparseDOK(self._zero)
        mat._items = {key: value * other for key, value in self._items.items()}
        return mat

    def _mul_matrix(self, other: MatrixSparse) -> MatrixSparse:
        """Multiply the matrix by another matrix

        Args:
            other (MatrixSparse): the matrix to multiply by

        Returns:
            MatrixSparseDOK: the matrix with the other matrix multiplied
        """
        if not isinstance(other, MatrixSparseDOK):
            raise ValueError("_mul_matrix() invalid arguments")
        if self.zero != other.zero:
            raise ValueError("_mul_matrix() incompatible matrices")

        dim1 = self.dim()
        dim2 = other.dim()
        size1_x = dim1[1][0] - dim1[0][0] + 1 #M1 rows dimension
        size1_y = dim1[1][1] - dim1[0][1] + 1 #M1 col dimension
        size2_x = dim2[1][0] - dim2[0][0] + 1 #M2 rows dimension
        size2_y = dim2[1][1] - dim2[0][1] + 1 #M2 col dimension

        if size1_y != size2_x:
          raise ValueError("_mul_matrix() incompatible matrices")

        min_row_m1 = dim1[0][0]
        min_col_m1 = dim1[0][1]
        min_row_m2 = dim2[0][0]
        min_col_m2 = dim2[0][1]

        mat = MatrixSparseDOK(self._zero)

        dic = {}
        #very cringe loops o meu cerebro esta a morrer apos ter que calcular isto mas funciona sempre creio eu
        for i in range(size1_x):
            for j in range(size2_y):
                for k in range(size2_x):
                    dic.update({(i+min_row_m1,j+min_col_m2): dic.get((i+min_row_m1,j+min_col_m2),0) 
                    + self._items.get((i+min_row_m1,k+min_col_m1),0)*other._items.get((k+min_row_m2,j+min_col_m2),0)})

        mat._items = dic
        return mat
        
  

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
        if not isinstance(size, int) or size < 0:
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

    def merge_two_lists(self, offset, list1, indexes1, list2, indexes2, zero) -> tuple[int, list, list]:
        """Merge two lists of indexes

        Args:
            offset (int): the offset to add to the indexes
            list1 (list): the first list of indexes
            indexes1 (list): the indexes of the first list
            list2 (list): the second list of indexes
            indexes2 (list): the indexes of the second list
            zero (float): the value of the zero elements

        Returns:
            tuple[int, list, list]: the offset, the merged list and the merged indexes
        """
        if offset >= 0:
            list_1 = [1 if x != zero else 0 for x in list1] + [0] * offset
            list_2 = [0] * offset + [2 if x != zero else 0 for x in list2]
        else:
            list_1 = [1 if x != zero else 0 for x in list1]
            list_2 = [2 if list2[x-offset] != zero else 0 for x in range(len(list2) + offset)]
            list_2 = list_2 + [0] * (-offset)
        list_sum = [x + y for x, y in zip(list_1, list_2)]
        if 3 in list_sum:
            return self.merge_two_lists(offset + 1, list1, indexes1, list2, indexes2, zero)
        else:
            merged_list = []
            indexes = []
            for i in range(len(list_sum)):
                if list_sum[i] == 1:
                    merged_list.append(list1[i])
                    indexes.append(indexes1[i])
                elif list_sum[i] == 2:
                    merged_list.append(list2[i-offset])
                    indexes.append(indexes2[i-offset])
                elif list_sum[i] == 0:
                    merged_list.append(zero)
                    indexes.append(-1)
            return offset, merged_list, indexes

    def order_by_density(self, list_of_lists, indexes, zero) -> tuple[list, list]:
        """Order a list of lists by density of the values not equal to zero

        Args:
            list_of_lists (list): the list of lists to order
            indexes (list): the list of indexes of the lists
            zero (float): the value of the zero elements

        Returns:
            tuple[list, list]: the ordered list of lists and the indexes of the lists
        """
        # measure the density of zeros in each list
        zeros = [lis.count(zero) for lis in list_of_lists]
        # order the list zeros and keep the orignal indexes
        rows = sorted(list_of_lists, key=lambda x: x.count(zero))
        # zeros_ordered = [x for _, x in sorted(zip(zeros, list_of_lists))]
        # order the indexes by the zeros
        indexes_ordered = [x for _, x in sorted(zip(zeros, indexes))]
        return rows, indexes_ordered
        
    def compress(self) -> compressed:
        """Compress the matrix

        Raises:
            ValueError: if the matrix is to dense

        Returns:
            compressed: the compressed matrix
        """
        if self.sparsity() < 0.5:
            raise ValueError("compress() dense matrix")
        zero = self._zero
        dim = self.dim()
        upper_left_pos = dim[0]
        # get all the rows in the dim
        rows = [[self.row(row)._items.get((row,col),zero) for col in range(dim[0][1], dim[1][1]+1)] for row in range(dim[0][0], dim[1][0]+1)]
        indexes = [[i]*len(rows[0]) for i in range(dim[0][0], dim[1][0]+1)]
        # order the rows by density of zeros
        rows_ordered, indexes_ordered = self.order_by_density(rows, indexes, zero)
        # insert first row and index
        merged_rows = rows_ordered[0]
        merged_indexes = indexes_ordered[0]
        offsets = []
        # calculate the first offset
        for val in merged_rows:
            if val != zero:
                offsets.append(merged_rows.index(val))
                break
        for row, index in zip(rows_ordered, indexes_ordered):
            # skip first row (already added)
            if rows_ordered.index(row) != 0:
                # skip rows with all zeros
                if not all(val == zero for val in row):
                    # get the collum of the first non zero value in list2
                    for val in row:
                        if val != zero:
                            firstcol = row.index(val)
                    offset, merged_rows, merged_indexes = self.merge_two_lists(-firstcol, merged_rows, merged_indexes,row, index, zero)
                else:
                    offset = 0
                # if cant add in the list add in front
                if offset == len(merged_rows):
                    merged_rows += row
                    merged_indexes += index
                offsets.append(offset)
        for i, val in enumerate(reversed(merged_rows)):
                if val != zero:
                    break
                y = len(merged_rows) - i - 1
                merged_rows.pop(y)
                merged_indexes.pop(y)
        # order offsets by the index of the row
        offsets_ordered = [x for _, x in sorted(zip(indexes_ordered, offsets))]
        return (upper_left_pos, zero, tuple(merged_rows), tuple(merged_indexes), tuple(offsets_ordered)) 

    @staticmethod
    def doi(compressed_vector: compressed, pos: Position) -> float:
        """Get the value of the element at the given position

        Args:
            compressed_vector (compressed): the compressed vector
            pos (Position): the position of the element

        Returns:
            float: the value of the element at the given position
        """
        if not isinstance(compressed_vector, tuple):
            raise ValueError("doi() invalid parameters")
        if len(compressed_vector) != 5:
            raise ValueError("doi() invalid parameters")
        if not isinstance(compressed_vector[0], tuple):
            raise ValueError("doi() invalid parameters")
        if len(compressed_vector[0]) != 2:
            raise ValueError("doi() invalid parameters")
        if not isinstance(compressed_vector[1], float):
            raise ValueError("doi() invalid parameters")
        if not isinstance(compressed_vector[2], tuple):
            raise ValueError("doi() invalid parameters")
        if not isinstance(compressed_vector[3], tuple):
            raise ValueError("doi() invalid parameters")
        if len(compressed_vector[2]) != len(compressed_vector[3]):
            raise ValueError("doi() invalid parameters")
        if not isinstance(compressed_vector[4], tuple):
            raise ValueError("doi() invalid parameters")
        if not isinstance(pos, Position):
            if not isinstance(
            Position.convert_to_pos(Position, pos), Position
            ) and not isinstance(pos, Position):
                raise ValueError("doi() invalid arguments")
            pos = Position.convert_to_pos(Position, pos)
        upper_left_pos = compressed_vector[0]
        zero = compressed_vector[1]
        value_rows = compressed_vector[2]
        value_indexes = compressed_vector[3]
        offsets = compressed_vector[4]
        offsets = [upper_left_pos[1] - off for off in offsets]
        indexes_sorted = sorted(list(set(value_indexes)))
        if -1 in indexes_sorted:
            indexes_sorted.remove(-1)
        offsets_dic = {ind: offsets[i] for i,ind in enumerate(range(indexes_sorted[0], indexes_sorted[-1]+1))}
        index_value_dic = {(index, col+offsets_dic.get(index)): value for col, (index, value) in enumerate(zip(value_indexes, value_rows)) if index != -1}
        return index_value_dic.get(pos, zero)

    @staticmethod
    def decompress(compressed_vector: compressed) -> MatrixSparseDOK:
        """Decompress the matrix

        Args:
            compressed_vector (compressed): the compressed matrix

        Returns:
            MatrixSparseDOK: the decompressed matrix
        """
        if not isinstance(compressed_vector, tuple):
            raise ValueError("decompress() invalid parameters")
        if len(compressed_vector) != 5:
            raise ValueError("decompress() invalid parameters")
        if not isinstance(compressed_vector[0], tuple):
            raise ValueError("decompress() invalid parameters")
        if len(compressed_vector[0]) != 2:
            raise ValueError("decompress() invalid parameters")
        if not isinstance(compressed_vector[1], float):
            raise ValueError("decompress() invalid parameters")
        if not isinstance(compressed_vector[2], tuple):
            raise ValueError("decompress() invalid parameters")
        if not isinstance(compressed_vector[3], tuple):
            raise ValueError("decompress() invalid parameters")
        if len(compressed_vector[2]) != len(compressed_vector[3]):
            raise ValueError("decompress() invalid parameters")
        if not isinstance(compressed_vector[4], tuple):
            raise ValueError("decompress() invalid parameters")

        upper_left_pos = compressed_vector[0]
        zero = compressed_vector[1]
        value_rows = compressed_vector[2]
        value_indexes = compressed_vector[3]
        offsets = compressed_vector[4]
        offsets = [upper_left_pos[1] - off for off in offsets]
        # Create a new matrix
        matrix = MatrixSparseDOK(zero)
        # Create a list merging the values with the indexes
        index_value = [((index, col), value) for col, (index, value) in enumerate(zip(value_indexes, value_rows))]
        # list of indexes sorted no repetitions 
        indexes_sorted = sorted(list(set(value_indexes)))
        if -1 in indexes_sorted:
            indexes_sorted.remove(-1)
        ind_off = [(ind,offsets[i]) for i, ind in enumerate(range(indexes_sorted[0], indexes_sorted[-1]+1))]
        for index, offset in ind_off:
            # get the values of the row
            values = [(position, val) for position, val in index_value if position[0] == index]
            # insert the values in the matrix
            for position, value in values:
                if position[0] != -1:
                    matrix[index, (position[1]+offset)] = value
        return matrix

