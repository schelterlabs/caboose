import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import caboose
#from .nbr_base import NBRBase
import math

from nbr_base import NBRBase

class TIFUKNN(NBRBase):
    def __init__(self, train_baskets, test_baskets,valid_baskets,basket_count_min=0,min_item_count =5 ,
                 k = 300, m = 7, rb = 1, rg = 0.6, alpha = 0.7):
        super().__init__(train_baskets,test_baskets,valid_baskets,basket_count_min)
        self.k = k
        self.m = m
        self.rb = rb
        self.rg = rg
        self.alpha = alpha
        self.min_item_count = min_item_count

        self.user_keys = []
        self.user_reps = []
        self.nn_indices = None
        self.item_id_mapper = {}
        self.id_item_mapper = {}

    def train(self):
        print('initial data processing')
        all_items = self.train_baskets[['item_id']].drop_duplicates()['item_id'].tolist()
        all_users = self.train_baskets[['user_id']].drop_duplicates()['user_id'].tolist()
        item_counts = self.train_baskets.groupby(['item_id']).size().to_frame(name = 'item_count').reset_index()
        item_counts = item_counts[item_counts['item_count']> self.min_item_count]
        item_counts_dict = dict(zip(item_counts['item_id'],item_counts['item_count']))
        print("item count:",len(item_counts_dict))
        counter = 0
        for i in range(len(all_items)):
            if all_items[i] in item_counts_dict:
                self.item_id_mapper[all_items[i]] = counter
                self.id_item_mapper[counter] = all_items[i]
                counter+=1

        sorted_baskets = self.train_baskets.sort_values(['user_id','date'])
        sorted_baskets = sorted_baskets[['user_id','basket_id']].drop_duplicates()
        user_baskets_df = sorted_baskets.groupby('user_id')['basket_id'].apply(list).reset_index()
        user_baskets_dict = dict(zip(user_baskets_df['user_id'],user_baskets_df['basket_id']))

        basket_items_df = self.train_baskets[['basket_id','item_id']].drop_duplicates().groupby('basket_id')['item_id'] \
            .apply(list).reset_index()
        basket_items_dict = dict(zip(basket_items_df['basket_id'],basket_items_df['item_id']))


        user_keys = []
        user_reps = []
        print('compute user reps',len(all_users))
        counter = 0
        for user in all_users:
            counter+=1
            if counter % 100 == 0:
                print(counter,' users passed')
            rep = np.array([0.0]* len(self.item_id_mapper))

            baskets = user_baskets_dict[user]
            group_size = math.ceil(len(baskets)/self.m)
            addition = (group_size * self.m) - len(baskets)

            basket_groups = []
            basket_groups.append(baskets[:group_size-addition])
            for i in range(self.m-1):
                basket_groups.append(baskets[group_size-addition+(i* group_size):group_size-addition+((i+1)* group_size)])

            for i in range(self.m):
                group_rep = np.array([0.0]* len(self.item_id_mapper))
                for j in range(1,len(basket_groups[i])+1):
                    basket = basket_groups[i][j-1]

                    rep = [0]* len(self.item_id_mapper)
                    for item in basket_items_dict[basket]:
                        if item in self.item_id_mapper:
                            rep[self.item_id_mapper[item]] = 1

                    basket_rep = np.array(rep) * math.pow(self.rb, group_size-j)
                    group_rep += basket_rep
                group_rep /= group_size

                rep += np.array(group_rep) * math.pow(self.rg, self.m-i)

            rep /= self.m
            user_reps.append(rep)
            user_keys.append(user)

        self.user_keys = user_keys
        self.user_reps = np.array(user_reps)

        representations = csr_matrix(self.user_reps)
        num_rows, num_cols = representations.shape
        print(representations.shape)
        print('start of knn')
        caboose.Index(num_rows, num_cols, representations.indptr,
                      representations.indices, representations.data,
                      self.k)
        print('knn finished')

    def predict(self):
        ret_dict = {}
        for i in range(len(self.user_keys)):
            user = self.user_keys[i]
            user_rep = self.user_reps[i]

            user_nns = self.nn_indices[i].tolist()[1:]

            nn_rep = np.array([0.0]* len(user_rep))
            for neighbor in user_nns:
                nn_rep += self.user_reps[neighbor]
            nn_rep /= len(user_nns)

            final_rep = (user_rep * self.alpha + (1-self.alpha) * nn_rep).tolist()
            final_rep_sorted = sorted(range(len(final_rep)), key=lambda k: final_rep[k], reverse=True)

            top_items = []
            for item_index in final_rep_sorted:
                top_items.append(self.id_item_mapper[item_index])
            ret_dict[user] = top_items
        return ret_dict


train_baskets = pd.read_csv('data/instacart_30k/train_baskets.csv')
test_baskets = pd.read_csv('data/instacart_30k/test_baskets.csv')
valid_baskets = pd.read_csv('data/instacart_30k/valid_baskets.csv')
tifu = TIFUKNN(train_baskets, test_baskets, valid_baskets)
tifu.train()