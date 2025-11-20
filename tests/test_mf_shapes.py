from src.model.matrix_factorization import MatrixFactorization
import numpy as np

def test_mf_shapes():
    mf = MatrixFactorization(5,7)
    assert mf.U.shape==(5,20)
    assert mf.V.shape==(7,20)
