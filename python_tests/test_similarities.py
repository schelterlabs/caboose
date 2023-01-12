import caboose
import numpy as np
from scipy.sparse import csr_matrix


def dense_cosine(A):
    similarity = np.dot(A, A.T)
    square_mag = np.diag(similarity)
    inv_square_mag = 1 / square_mag
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    inv_mag = np.sqrt(inv_square_mag)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    np.fill_diagonal(cosine, 0)
    return cosine


def caboose_index(representations, k):
    num_rows, num_cols = representations.shape
    return caboose.Index(num_rows, num_cols, representations.indptr,
                         representations.indices, representations.data,
                         k)


def compare(row, cosine, index, k):
    max_indices = cosine[row,:].argsort()[-k:][::-1]
    max_values = cosine[row,max_indices]
    from_numpy = dict(zip(max_indices, max_values))
    from_numpy = {index:value for index,value in from_numpy.items() if value != 0}
    from_caboose = dict((index, value) for index, value in index.topk(row))

    assert len(from_numpy) == len(from_caboose)
    for index, value in from_caboose.items():
        assert index in from_numpy
        assert abs(value - from_numpy[index]) < 0.0001


def run_comparison(A, k):
    A_sparse = csr_matrix(A)
    cosine = dense_cosine(A)
    index = caboose_index(A_sparse, k)

    for row in range(0, A.shape[0]):
        compare(row, cosine, index, k)


def test_mini_example():
    A = np.array(
        [[1, 1, 1, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 0, 1],
         [0, 0, 0, 1, 0]],
        dtype='float64')

    run_comparison(A, 1)
    run_comparison(A, 2)
    run_comparison(A, 3)


def test_random_matrices():
    from scipy.sparse import random
    from scipy import stats
    from numpy.random import default_rng

    rng = default_rng()

    for _ in range(0, 5):
        rvs = stats.poisson(25, loc=10).rvs
        S = random(10, 20, density=0.25, random_state=rng, data_rvs=rvs, format='csr')

        run_comparison(S.A, 2)
        run_comparison(S.A, 5)


def test_mini_example_with_forgetting():
    A = np.array(
        [[1, 1, 1, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 0, 1],
         [0, 0, 0, 1, 0]],
        dtype='float64')

    k = 2
    A_sparse = csr_matrix(A)
    index = caboose_index(A_sparse, k)

    A[0, 1] = 0.0
    cosine = dense_cosine(A)
    index.forget(0, 1)

    for row in range(0, A.shape[0]):
        compare(row, cosine, index, k)

    A[1, 3] = 0.0
    cosine = dense_cosine(A)
    index.forget(1, 3)

    print("\n")
    print(cosine)
    print("----")
    for row in range(0, 4):
        print(row, '-->', index.topk(row))

    for row in range(0, A.shape[0]):
        compare(row, cosine, index, k)
