import caboose
import numpy as np
from scipy.sparse import csr_matrix

def caboose_index(representations, k):
    num_rows, num_cols = representations.shape
    return caboose.Index(num_rows, num_cols, representations.indptr,
                         representations.indices, representations.data,
                         k)


row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

representations = csr_matrix((data, (row, col)), shape=(3, 3))

index = caboose_index(representations, 1)

print(index.topk(0))
#print(caboose.receive_csr_matrix(3, 3, A.indptr, A.indices, A.data))

