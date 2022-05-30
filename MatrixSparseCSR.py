from __future__ import annotations
from functools import reduce
from MatrixSparse import *
from Position import *
from typing import Union


class MatrixSparseCSR(MatrixSparse):
    # V -> Value matrix row, column order
    # Col_index -> Column number for every value
    # Row_count -> Count of the number of values per row
    def __init__(self, zero: float = 0):
        """Create a new sparse matrix with Dictionary of Keys (DOK)

        Args:
            zero (float): value to be used for zero elements
        """
        if not isinstance(zero, (float, int)):
            raise ValueError("__init__() invalid arguments")
        super(MatrixSparseCSR, self).__init__(zero)
        self._values = []
        self._col_index = []
        self._row_count = []

    @MatrixSparse.zero.setter
    def zero(self, val: Union[int, float]):
        """Set the value to be used for zero elements

        Args:
            val (Union[int, float]): value to be used for zero elements
        """
        MatrixSparse.zero.fset(self, val)
        for key in self:
            if self[key] == self.zero:
                self.del_pos(key)
        

    def __copy__(self):
        """Create a copy of the matrix

        Returns:
            MatrixSparseCSR: a copy of the matrix
        """
        copy = MatrixSparseCSR(self.zero)
        copy._values = self._values
        copy._col_index = self._col_index
        copy._row_count = self._row_count
        return copy

    def __eq__(self, other: MatrixSparseCSR):
        """Compare two sparse matrices

        Args:
            other (MatrixSparseCSR): the matrix to compare with

        Raises:
            ValueError: if the other matrix is not a sparse matrix

        Returns:
            bool: True if the two matrices are equal, False otherwise
        """
        if not isinstance(other, MatrixSparseCSR):
            return False
        return self._values == other._values and self._col_index == other._col_index and self._row_count == other._row_count
        

    def __iter__(self):
        """Create an iterator for the matrix

        Returns:
            MatrixSparseCSR: an iterator for the matrix
        """
        rows = [i for i in range(self._row_count[0],self._row_count[0] + len(self._row_count[1:]))]
        #rows_cols = [[(row, col) for col in self.row(row)._col_index] for row in rows]
        rows_cols = []
        
        for row in rows:
            num_of_vals_in_row = self._row_count[(row - self._row_count[0]) + 1] - self._row_count[row-self._row_count[0]]
            available_cols = self._col_index[row-self._row_count[0]:(row-self._row_count[0])+num_of_vals_in_row]
            for col in available_cols:
                rows_cols.append((row, col))
        self._iterator = iter(rows_cols)
        return self._iterator

    def __next__(self):
        """Get the next element of the matrix

        Returns:
            tuple[Position, float]: the next element of the matrix
        """
        return self._iterator.next()

    def del_pos(self, pos: Union[Position, position]):
        """Delete the element at the given position

        Args:
            pos (Union[Position, position]): the position of the element to delete
        """
        if not isinstance(pos, Position):
            pos = Position.convert_to_pos(Position, pos)
        sel_r = pos[0]
        sel_c = pos[1]
        if sel_r < self._row_count[0] or sel_r > self._row_count[0] + len(self._row_count[1:]):
            return
        available_cols = self._col_index[self._row_count[sel_r]:self._row_count[sel_r+1]]
        if not (sel_c in available_cols):
            return
        index = self._row_count[sel_r] + available_cols.index(sel_c)
        self._values.pop(index)
        self._col_index.pop(index)
        for row in range(sel_r, len(self._row_count)):
            self._row_count[row] -= 1
        


    def __getitem__(self, pos: Union[Position, position]) -> float:
        """Get the value of the element at the given position

        Args:
            pos (Union[Position, position]): the position of the element

        Raises:
            ValueError: if the position is not valid

        Returns:
            float: the value of the element at the given position
        """
        
        if not isinstance(pos, Position):
            if not isinstance(
            Position.convert_to_pos(Position, pos), Position
        ) and not isinstance(pos, Position):
                raise ValueError("__getitem__() invalid arguments")
            pos = Position.convert_to_pos(Position, pos)
        
        sel_r = pos[0]
        sel_c = pos[1]
        if self._row_count == []:
            return self._zero
        if sel_r < self._row_count[0] or sel_r > self._row_count[0] + len(self._row_count[1:]):
            return self._zero
        if sel_r > len(self._row_count) - 1:
            return self._zero
        num_of_vals_in_row = self._row_count[(sel_r - self._row_count[0]) + 1] - self._row_count[sel_r-self._row_count[0]]
        available_cols = self._col_index[sel_r-self._row_count[0]:(sel_r-self._row_count[0])+num_of_vals_in_row]
        if not (sel_c in available_cols):
            return self._zero
        return self._values[sel_r-self._row_count[0] + available_cols.index(sel_c)]

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
            return
        if self.__getitem__(pos) != self._zero:
            return
        sel_r = pos[0]
        sel_c = pos[1]
        if len(self._values) == 0:
            self._values.append(val)
            self._col_index.append(sel_c)
            self._row_count.append(sel_r)
            self._row_count.append(sel_r + 1)
            return       
        initial_row = self._row_count[0]
        distance_to_first_row = sel_r - initial_row
        # If the row is not in the matrix, add it
        # before the first row
        if distance_to_first_row < 0:
            self._row_count.insert(0, sel_r)
            self._col_index.insert(0, sel_c)
            self._values.insert(0, val)
            return
        # after the last row
        if distance_to_first_row >= len(self._row_count) - 1:
            self._row_count.append(self._row_count[-1]+1)
            self._col_index.append(sel_c)
            self._values.append(val)
            return
        #determine the number of values in the row
        num_of_vals_in_row = self._row_count[(sel_r - self._row_count[0]) + 1] - self._row_count[sel_r-self._row_count[0]]
        available_cols = self._col_index[sel_r-self._row_count[0]:(sel_r-self._row_count[0])+num_of_vals_in_row]
        if sel_c < available_cols[0]:
            self._col_index.insert(distance_to_first_row, sel_c)
            self._values.insert(distance_to_first_row, val)
        elif sel_c > available_cols[-1]:
            self._col_index.insert(distance_to_first_row + num_of_vals_in_row, sel_c)
            self._values.insert(distance_to_first_row + num_of_vals_in_row, val)
        # If the row is in the matrix, add the value
        for row in range(distance_to_first_row + 1, len(self._row_count)):
            self._row_count[row] += 1
        return


    def __len__(self) -> int:
        """Get the number of elements in the matrix

        Returns:
            int: the number of elements in the matrix
        """
        return len(self._values)

    def _add_number(self, other: Union[int, float]) -> Matrix:
        """Add a number to the matrix

        Args:
            other (Union[int, float]): the number to add

        Raises:
            ValueError: if the other number is not a number

        Returns:
            MatrixSparseCSR: the matrix with the number added
        """
        if not isinstance(other, (int, float)):
            raise ValueError("_add_number: invalid arguments")
        mat = MatrixSparseCSR(self._zero)
        for pos in self:
            mat[pos] = self[pos] + other
        return mat        

    def _add_matrix(self, other: MatrixSparseCSR) -> MatrixSparseCSR:
        """Add a matrix to the matrix

        Args:
            other (MatrixSparseCSR): the matrix to add

        Returns:
            MatrixSparseCSR: the matrix with the other matrix added
        """
        if not isinstance(other, MatrixSparseCSR):
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
        mat = MatrixSparseCSR(self.zero)
        # Works?
        for pos in self:
            mat[pos] = self[pos]
        for pos in other:
            mat[pos] += other[pos]
        return mat

    def _mul_number(self, other: Union[int, float]) -> Matrix:
        """Multiply the matrix by a number

        Args:
            other (Union[int, float]): the number to multiply by

        Raises:
            ValueError: if the other number is not a number

        Returns:
            MatrixSparseCSR: the matrix with the number multiplied
        """
        if not isinstance(other, (int, float)):
            raise ValueError("_mul_number: invalid arguments")
        mat = MatrixSparseCSR(self._zero)
        for pos in self:
            mat[pos] = self[pos] * other
        return mat

    def _mul_matrix(self, other: MatrixSparse) -> MatrixSparse:
        """Multiply the matrix by another matrix

        Args:
            other (MatrixSparse): the matrix to multiply by

        Returns:
            MatrixSparseCSR: the matrix with the other matrix multiplied
        """
        if not isinstance(other, MatrixSparseCSR):
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

        mat = MatrixSparseCSR(self._zero)
        # Works?
        for pos in self:
            row = pos[0]
            col = pos[1]
            val = self[pos]
            for pos2 in other:
                row2 = pos2[0]
                col2 = pos2[1]
                val2 = other[pos2]
                if col == col2:
                    mat[(row + min_row_m1, col + min_col_m2)] += val * val2
        return mat
        
  

    def dim(self) -> tuple[Position, ...]:
        """Get the dimensions of the matrix

        Returns:
            tuple[Position, ...]: the dimensions of the matrix
        """
        if len(self) == 0:
            return tuple()
        return (Position(self._row_count[0], min(self._col_index)),
                Position(self._row_count[0] + (len(self._row_count[1:])-1), max(self._col_index)))
        

    def row(self, row: int) -> MatrixSparseCSR:
        """Get the row of the matrix

        Args:
            row (int): the row to get

        Raises:
            ValueError: if the row is not valid

        Returns:
            MatrixSparseCSR: the row of the matrix
        """
        if not isinstance(row, int):
            raise ValueError("spmatrix_row: invalid arguments")
        mat = MatrixSparseCSR(self._zero)
        for key in self:
            if key[0] == row:
                mat[key] = self[key]
        return mat

    def col(self, col: int) -> Matrix:
        """Get the column of the matrix

        Args:
            col (int): the column to get

        Raises:
            ValueError: if the column is not valid

        Returns:
            MatrixSparseCSR: the column of the matrix
        """
        if not isinstance(col, int):
            raise ValueError("spmatrix_col: invalid arguments")
        mat = MatrixSparseCSR(self._zero)
        for key in self:
            if key[1] == col:
                mat[key] = self[key]
        return mat

    def diagonal(self) -> Matrix:
        """Get the diagonal of the matrix

        Returns:
            MatrixSparseCSR: the diagonal of the matrix
        """
        if not (self.square()):
            raise ValueError("spmatrix_diagonal: matrix not square")
        mat = MatrixSparseCSR(self._zero)
        for key in self:
            if key[0] == key[1]:
                mat[key] = self[key]
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
    def eye(size: int, unitary: float = 1.0, zero: float = 0.0) -> MatrixSparseCSR:
        """Create an identity matrix

        Args:
            size (int): the size of the matrix
            unitary (float): the value of the unitary elements
            zero (float): the value of the zero elements

        Returns:
            MatrixSparseCSR: the identity matrix
        """
        if not isinstance(size, int) or size < 0:
            raise ValueError("eye() invalid parameters")
        if not isinstance(unitary, (int, float)) or not isinstance(zero, (int, float)):
            raise ValueError("eye() invalid parameters")
        mat = MatrixSparseCSR(zero)
        for i in range(size):
            mat[i, i] = unitary
        return mat

    def transpose(self) -> MatrixSparseCSR:
        """Transpose the matrix

        Returns:
            MatrixSparseCSR: the transpose of the matrix
        """
        mat = MatrixSparseCSR(self._zero)
        for key in self:
            mat[key[1], key[0]] = self[key]
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
        pass
        if self.sparsity() < 0.5:
            raise ValueError("compress() dense matrix")
        zero = self._zero
        dim = self.dim()
        upper_left_pos = dim[0]
        # get all the rows in the dim
        #rows = [self[row,col] for row, col in zip(range(dim[0][0], dim[1][0]+1), range(dim[0][1], dim[1][1]+1))]
        rows = []
        for row in range(dim[0][0], dim[1][0]+1):
            cur_row = []
            for col in range(dim[0][1], dim[1][1]+1):
                cur_row.append(self[row, col])
            rows.append(cur_row)
        indexes = []
        for i,row_s in zip(range(dim[0][0], dim[1][0]+1), rows):
            indexes.append([i]*len(row_s))
        #indexes = [[i]*len(row[0]) for i in range(dim[0][0], dim[1][0]+1)]
        # order the rows by density of zeros
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
    def decompress(compressed_vector: compressed) -> MatrixSparseCSR:
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
        matrix = MatrixSparseCSR(zero)
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

