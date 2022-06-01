from __future__ import annotations
from MatrixSparse import *
from Position import *

class Import():
    """
    Class for importing data from a file.
    """
    def __init__(self, filename: str):
        """Initialize the class with the filename.
        
        Args:
            filename(str): The filename to import from.
        """
        self.filename = filename
    def import_matrix(self, m_type) -> MatrixSparse:
        """Import the matrix from a .mm file
        
        Returns:
            MatrixSparse: The imported matrix
        """
        with open(self.filename, 'r') as f:
            line = f.readline()
            while line[0] == '%':
                line = f.readline()
            line = line.split()
            row_n = int(line[0])
            col_n = int(line[1])
            number_elements = int(line[2])
            zero = int(line[3])
            matrix = m_type(zero)
            for _ in range(number_elements):
                line = f.readline()
                line = line.split()
                row = int(line[0]) - 1
                col = int(line[1]) - 1
                value = float(line[2])
                matrix[(row, col)] = value
        return matrix
        

class Export():
    """
    Generate a .mm file from a matrix
    """
    def __init__(self, filename: str):
        """Initialize the class

        Args:
            filename (str): filename of the .mm file
        """
        self.filename = filename
    def export_matrix(self, matrix: MatrixSparse):
        """Export the matrix to a .mm file
        
        Args:
            matrix (MatrixSparse): matrix to export
        """
        with open(self.filename, 'w') as f:
            f.write("%%MatrixMarket matrix coordinate real general\n")
            dim = matrix.dim()
            row_n = dim[0][1] - dim[0][0]
            col_n = dim[1][1] - dim[1][0]
            number_elements = len(matrix)
            zero = matrix.zero
            f.write(f"  {row_n}  {col_n}  {number_elements}  {zero}\n")
            for key in matrix:
                row = key[0] + 1
                col = key[1] + 1
                value = matrix[key]
                f.write(f"    {row}     {col}   {value}\n")

