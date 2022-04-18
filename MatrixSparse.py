from __future__ import annotations
from Matrix import *

position = tuple[int, int]
compressed = tuple[position, float, tuple[float], tuple[int], tuple[int]]

class MatrixSparse(Matrix):
    _zero = float

    def __init__(self, zero: float = 0):
        if not isinstance(zero, (float,int)):
            raise TypeError("__init__: invalid arguments")
        self._zero = zero

    @property
    def zero(self) -> float:
        return self._zero

    @zero.setter
    def zero(self, val: Union[int, float]):
        if not isinstance(val, (int, float)):
            raise TypeError('zero(): invalid arguments')
        if val == self.zero:
            return
        self._zero = val

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def sparsity(self) -> float:
        """Compute the sparsity of the matrix

        Returns:
            float: sparsity percentage of the matrix (between 0 and 1)
        """
        dimension = self.dim()
        if len(self) == 0:
            return 1
        if len(self) == 1:
            return 0
        all_values = (dimension[1][1] - dimension[0][1] + 1) * \
            (dimension[1][0] - dimension[0][0] + 1)
        zero_values = all_values - len(self)
        return float(zero_values/all_values)

    @staticmethod
    @abstractmethod
    def eye(size: int, unitary: float = 1.0, zero: float = 0.0) -> MatrixSparse:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compress(self) -> compressed:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def doi(compressed_vector: compressed, pos: Position) -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def decompress(compressed_vector: compressed) -> Matrix:
        raise NotImplementedError
