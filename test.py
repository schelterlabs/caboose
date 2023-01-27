import caboose
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import load_npz
import similaripy as sim
from datetime import datetime


def caboose_index(representations, k):
    num_rows, num_cols = representations.shape
    return caboose.Index(num_rows, num_cols, representations.indptr,
                         representations.indices, representations.data, k)

def compare(npz_file, k, num_repetitions):

    print(f'# Running comparison for {npz_file}')
    A = load_npz(npz_file)

    num_rows, num_columns = A.shape
    nnz = A.nnz

    for _ in range(0, num_repetitions):

        start_time = datetime.now()
        _ = sim.cosine(A, k=k)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print(f'INDEX_BUILDING,{npz_file},{num_rows},{num_columns},{nnz},{k},similaripy,{duration}')

        start_time = datetime.now()
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine', radius=None, n_jobs=1)
        nbrs = nbrs.fit(A)
        _ = nbrs.kneighbors(A)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print(f'INDEX_BUILDING,{npz_file},{num_rows},{num_columns},{nnz},{k},sklearn,{duration}')

        start_time = datetime.now()
        _ = caboose_index(A, k)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print(f'INDEX_BUILDING,{npz_file},{num_rows},{num_columns},{nnz},{k},caboose,{duration}')


num_repetitions = 1
ks = [10, 50, 100]

for k in ks:
    compare("../caboose_index/spotify-raw.npz", ks, num_repetitions)
    #compare("tifu-instacart.npz", k, num_repetitions)
    #compare("../caboose_index/pernir-instacart.npz", k, num_repetitions)
    #compare("synthetic-10000-50000-0.02.npz", k, num_repetitions)
    #compare("../caboose_index/movielens10m-raw.npz", k, num_repetitions)
    #compare("../caboose_index/lastfm-raw.npz", k, num_repetitions)
    #compare("synthetic-100000-50000-0.01.npz", k, num_repetitions)
