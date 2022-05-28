import unittest
import sys
import copy
from MatrixSparseDOK import *

MatrixSparseImplementation = MatrixSparseDOK

class TestMatrixSparseMine(unittest.TestCase):
    def test_mytest(self):
        m = MatrixSparseImplementation()
        m[1,2] = 1.2
        m[2,0] = 2.0
        m[2,2] = 2.2
        result = m.compress()
        expected = ((1,0), 0 ,(2.0, 1.2, 2.2), (2,1,2), (-1,0))
        self.assertEqual(result, expected)
    def test_mytest2(self):
        m = MatrixSparseImplementation()
        pass