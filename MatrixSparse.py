from __future__ import annotations
from Matrix import *

position = tuple[int, int]
compressed = tuple[position, float, tuple[float], tuple[int], tuple[int]]

class MatrixSparse(Matrix):
    _zero = float

    def __init__(self, zero):
        if not isinstance(zero, float):
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
        for pos in self:
            if self[pos] == val:
                self[pos] = self._zero
        self._zero = val

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def sparsity(self) -> float:
        # TODO: implement this method: NAVES
        pass

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