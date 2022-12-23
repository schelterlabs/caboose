import caboose
from scipy.sparse import csr_matrix, load_npz
from datetime import datetime
import numpy as np

#np.random.seed(1234)

representations = load_npz('tifu-instacart.npz')
print(representations.shape, representations.nnz)

k=300
num_rows, num_cols = representations.shape

start_time = datetime.now()
index = caboose.Index(num_rows, num_cols, representations.indptr,
                      representations.indices, representations.data, k)
end_time = datetime.now()
print('--caboose: {}'.format((end_time - start_time).total_seconds() * 1000))

start_time = datetime.now()
index_to_sample_from = caboose.Index(num_rows, num_cols, representations.indptr,
                                     representations.indices, representations.data, 10000)
end_time = datetime.now()
print('--caboose: {}'.format(end_time - start_time))

row_indices, column_indices = representations.nonzero()
print(len(row_indices), len(column_indices))
offsets = np.random.choice(len(row_indices), 50)


for offset in offsets:
    row = row_indices[offset]
    col = column_indices[offset]
    start_time = datetime.now()
    index.forget(row, col)
    end_time = datetime.now()
    print('forgetting: {}, {}: {}'.format(row, col, (end_time - start_time).total_seconds() * 1000))
