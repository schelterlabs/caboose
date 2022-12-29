from pernir import PernirNonOptimized
import pandas as pd

train_baskets = pd.read_csv('data/instacart_30k/train_baskets.csv.gz')

pernir = PernirNonOptimized(300, 0.5, 0.5)
pernir._init_user_neighbors(train_baskets, 300)