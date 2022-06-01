import unittest
import sys
import copy
from MatrixSparseDOK import *
from MatrixSparseCSR import *




class TestMatrixSparseMine(unittest.TestCase):
    MatrixSparseImplementation = MatrixSparseDOK
    def test_mytest(self):
        m = self.MatrixSparseImplementation()
        m[1,2] = 1.2
        m[2,0] = 2.0
        m[2,2] = 2.2
        result = m.compress()
        expected = ((1,0), 0 ,(2.0, 1.2, 2.2), (2,1,2), (-1,0))
        self.assertEqual(result, expected)



class TestMatrixSparseCSR(unittest.TestCase):
    MatrixSparseImplementation = MatrixSparseCSR
    def test_mytest(self):
        m = self.MatrixSparseImplementation()
        m[1,2] = 1.2
        m[2,2] = 2.2
        m[2,0] = 2.0
        result = m[2,2]
        expected = 2.2
        self.assertEqual(m._values, [1.2,2.0,2.2])
        self.assertEqual(result, expected)
        self.assertEqual(1.2, m[1,2])
        self.assertEqual(2.0, m[2,0])
    def test_mytest2(self):
        m = self.MatrixSparseImplementation()
        self.assertEqual(m[1,0], 0.0)
        m[1,1] = 1.1
        self.assertEqual(m[2,2], 0.0)
        self.assertEqual(m.dim(), ((1,1),(1,1)))
    def test_mytest3(self):
        m = self.MatrixSparseImplementation()
        m[1,1] = 1.1
        m[0,0] = 9.9
        self.assertEqual(m[0,0], 9.9)
        self.assertEqual(m[1,1], 1.1)
        self.assertEquals(m[0,1], 0.0)
    def test_mytest4(self):
        m = self.MatrixSparseImplementation()
        m[1,1] = 1.1
        m[0,0] = 9.9
        self.assertEqual(m.dim(), ((0,0),(1,1)))
    def test_mytest5(self):
        m = self.MatrixSparseImplementation()
        m[1,2] = 1.2
        m[2,2] = 2.2
        m[2,0] = 2.0
        compressed = m.compress()
        self.assertEqual(compressed, ((1,0), 0 ,(2.0, 1.2, 2.2), (2,1,2), (-1,0)))
    def test_mytest6(self):
        m = self.MatrixSparseImplementation()
        m[1,2] = 1.2
        self.assertEqual(m[1,2], 1.2)
        m[1,2] = 1.1
        self.assertEqual(m[1,2], 1.1)
