import caboose
import numpy as np
from scipy.sparse import csr_matrix

from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng
import similaripy as sim

from datetime import datetime

def caboose_index(representations, k):
    num_rows, num_cols = representations.shape
    return caboose.Index(num_rows, num_cols, representations.indptr,
                         representations.indices, representations.data,
                         k)

# https://stackoverflow.com/questions/36135927/get-top-n-items-of-every-row-in-a-scipy-sparse-matrix
def max_n(row_data, row_indices, n):
    i = row_data.argsort()[-n:]
    top_values = row_data[i]
    top_indices = row_indices[i]
    return top_values, top_indices, i

#for sizes in [(1000,1000), (1000,10000), (10000, 10000), (10000, 50000)]:
for sizes in [(50000, 100000)]:
    print(sizes)
    for k in (10, 50, 100):
        print(" ", k)
        for _ in range(0, 3):
            rng = default_rng()

            rvs = stats.poisson(25, loc=10).rvs
            S = random(sizes[0], sizes[1], density=0.01, random_state=rng, data_rvs=rvs, format='csr')
            A = csr_matrix(S.A)

            # start_time = datetime.now()
            # similarity = A.dot(A.T)
            # square_mag = similarity.diagonal()
            # inv_square_mag = 1 / square_mag
            # inv_square_mag[np.isinf(inv_square_mag)] = 0
            # inv_mag = np.sqrt(inv_square_mag)
            # inv_mag = csr_matrix(inv_mag)
            #
            # cosine = similarity
            # # TODO implement this: https://stackoverflow.com/questions/49254111/row-division-in-scipy-sparse-matrix
            # #cosine = similarity * inv_mag
            # #cosine = cosine.T * inv_mag
            # #cosine.setdiag(0.0)
            #
            # cosine_ll = cosine.tolil()
            # for i in range(cosine_ll.shape[0]):
            #     d,r=max_n(np.array(cosine_ll.data[i]),np.array(cosine_ll.rows[i]),2)[:2]
            #     cosine_ll.data[i]=d.tolist()
            #     cosine_ll.rows[i]=r.tolist()
            #
            # end_time = datetime.now()
            # print('sci-py: {}'.format(end_time - start_time))

            start_time = datetime.now()
            model = sim.cosine(A, k=k)
            end_time = datetime.now()
            print('--similaripy: {}'.format(end_time - start_time))

            start_time = datetime.now()
            index = caboose_index(A, k)
            end_time = datetime.now()
            print('--caboose_nbr: {}'.format(end_time - start_time))
