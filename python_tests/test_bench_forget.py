import caboose
import numpy as np
from scipy.sparse import csr_matrix

from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng

from datetime import datetime

def caboose_index(representations, k):
    num_rows, num_cols = representations.shape
    return caboose.Index(num_rows, num_cols, representations.indptr,
                         representations.indices, representations.data,
                         k)


rng = default_rng()

rvs = stats.poisson(25, loc=10).rvs
S = random(10000, 10000, density=0.05, random_state=rng, data_rvs=rvs, format='csr')

A = csr_matrix(S.A)
k = 10

row_indices, column_indices = A.nonzero()

start_time = datetime.now()
index = caboose_index(A, k)
end_time = datetime.now()
print('caboose: {}'.format(end_time - start_time))

offsets = np.random.choice(len(row_indices), 10)

for offset in offsets:
    row = row_indices[offset]
    col = column_indices[offset]
    start_time = datetime.now()
    index.forget(row, col)
    end_time = datetime.now()
    print('forgetting: {}, {}: {}'.format(row, col, end_time - start_time))