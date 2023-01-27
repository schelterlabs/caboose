import sys
import numpy as np
from scipy.sparse import csr_matrix
import caboose
import math
from sklearn.neighbors import NearestNeighbors
from caboose_nbr.nbr_base import NBRBase


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class TIFUKNN(NBRBase):
    def __init__(self, train_baskets, test_baskets, valid_baskets,mode, distance_metric = 'minkowski' ,basket_count_min=0, min_item_count =5 ,
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
        self.caboose = None
        self.item_id_mapper = {}
        self.id_item_mapper = {}
        self.user_map = {}
        self.mode = mode
        self.distance_metric = distance_metric
        self.all_user_nns = {}
        eprint('initial data processing')
        self.all_items = self.train_baskets[['item_id']].drop_duplicates()['item_id'].tolist()
        self.all_users = self.train_baskets[['user_id']].drop_duplicates()['user_id'].tolist()
        self.item_counts = self.train_baskets.groupby(['item_id']).size().to_frame(name='item_count').reset_index()
        self.item_counts = self.item_counts[self.item_counts['item_count'] > self.min_item_count]
        self.item_counts_dict = dict(zip(self.item_counts['item_id'], self.item_counts['item_count']))
        eprint("item count:", len(self.item_counts_dict))
        
        counter = 0
        for i in range(len(self.all_items)):
            if self.all_items[i] in self.item_counts_dict:
                self.item_id_mapper[self.all_items[i]] = counter
                self.id_item_mapper[counter] = self.all_items[i]
                counter += 1
        

    def train(self):
        #sorted_baskets = self.train_baskets.sort_values(['user_id','date'])
        sorted_baskets = self.train_baskets.sort_values(['user_id', 'order_number'])
        sorted_baskets = sorted_baskets[['user_id','basket_id']].drop_duplicates()
        user_baskets_df = sorted_baskets.groupby('user_id')['basket_id'].apply(list).reset_index()
        user_baskets_dict = dict(zip(user_baskets_df['user_id'], user_baskets_df['basket_id']))

        basket_items_df = self.train_baskets[['basket_id', 'item_id']].drop_duplicates().groupby('basket_id')['item_id'] \
            .apply(list).reset_index()
        basket_items_dict = dict(zip(basket_items_df['basket_id'],basket_items_df['item_id']))
        eprint('compute basket reps')
        basket_reps = {}
        counter = 0
        for basket in basket_items_dict:
            counter+=1
            if counter % 10000 == 0:
                eprint(counter, ' baskets passed')
            rep = np.zeros(len(self.item_id_mapper))
            for item in basket_items_dict[basket]:
                if item in self.item_id_mapper:
                    rep[self.item_id_mapper[item]] = 1
            basket_reps[basket] = rep

        user_keys = []
        user_reps = []
        eprint('compute user reps', len(self.all_users))
        counter = 0
        for user in self.all_users:
            counter+=1
            if counter % 1000 == 0:
                eprint(counter, ' users passed')
            rep = np.zeros(len(self.item_id_mapper))

            baskets = user_baskets_dict[user]
            group_size = math.ceil(len(baskets)/self.m)
            addition = (group_size * self.m) - len(baskets)

            basket_groups = []
            basket_groups.append(baskets[:group_size-addition])
            for i in range(self.m-1):
                basket_groups.append(baskets[group_size-addition+(i* group_size):group_size-addition+((i+1)* group_size)])

            for i in range(self.m):
                group_rep = np.array([0.0]* len(self.item_id_mapper))
                for j in range(1, len(basket_groups[i])+1):
                    basket = basket_groups[i][j-1]
                    basket_rep = np.array(basket_reps[basket]) * math.pow(self.rb, group_size-j)
                    group_rep += basket_rep
                group_rep /= group_size

                rep += np.array(group_rep) * math.pow(self.rg, self.m-i)

            rep /= self.m
            user_reps.append(rep)
            user_keys.append(user)

        self.user_keys = user_keys
        self.user_reps = np.array(user_reps)
        self.user_map = dict(zip(user_keys,range(len(user_keys))))

        representations = csr_matrix(self.user_reps)

        num_rows, num_cols = representations.shape
        eprint(representations.shape)
        eprint('start of knn', )
        if self.mode == 'caboose':
            self.caboose = caboose.Index(num_rows, num_cols, representations.indptr,
                                         representations.indices, representations.data,
                                         self.k)
        if self.mode == 'sklearn':
            nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='brute',metric =self.distance_metric).fit(user_reps)
            distances, indices = nbrs.kneighbors(user_reps)
            self.nn_indices = indices
        eprint('knn finished')

        
    def forget_interactions(self,user_item_pairs):
        if self.mode == 'caboose':
            for user,item in user_item_pairs:
                if item in self.item_id_mapper and user in self.user_map:
                    self.user_reps[self.user_map[user]][self.item_id_mapper[item]] = 0
                    self.caboose.forget(self.user_map[user],self.item_id_mapper[item])
        if self.mode == 'sklearn':
            self.train_baskets['user_item'] = self.train_baskets.apply(lambda x: [x['user_id'],x['item_id']],axis = 1)
            self.train_baskets = self.train_baskets[~self.train_baskets['user_item'].isin(user_item_pairs)]
            self.train_baskets.drop('user_item', axis=1, inplace=True)
            self.train()

    def predict_for_user(self, user_key, how_many):
        i = self.user_keys.index(user_key)
        user = self.user_keys[i]
        user_rep = self.user_reps[i]

        nn_rep = np.zeros(len(user_rep))
        if self.mode == 'sklearn':
            user_nns = self.nn_indices[i].tolist()[1:]
            for neighbor in user_nns:
                nn_rep += self.user_reps[neighbor]
        if self.mode == 'caboose':
            user_nns = self.caboose.topk(i)
            for neighbor, _ in user_nns:
                nn_rep += self.user_reps[neighbor]
        self.all_user_nns[user] = user_nns
        if len(user_nns) > 0:
            nn_rep /= len(user_nns)

        final_rep = (user_rep * self.alpha + (1-self.alpha) * nn_rep).tolist()
        final_rep_sorted = sorted(range(len(final_rep)), key=lambda k: final_rep[k], reverse=True)

        top_items = []
        for item_index in final_rep_sorted[:how_many]:
            top_items.append(self.id_item_mapper[item_index])

        return top_items

    def predict(self):
        ret_dict = {}
        for i in range(len(self.user_keys)):
            #TODO refactor to call predict_for_user
            user = self.user_keys[i]
            user_rep = self.user_reps[i]
            
            nn_rep = np.array([0.0]* len(user_rep))
            if self.mode == 'sklearn':
                user_nns = self.nn_indices[i].tolist()[1:]
                for neighbor in user_nns:
                    nn_rep += self.user_reps[neighbor]
            if self.mode == 'caboose':
                user_nns = self.caboose.topk(i)
                for neighbor, _ in user_nns:
                    nn_rep += self.user_reps[neighbor]
            self.all_user_nns[user] = user_nns
            nn_rep /= len(user_nns)

            final_rep = (user_rep * self.alpha + (1-self.alpha) * nn_rep).tolist()
            final_rep_sorted = sorted(range(len(final_rep)), key=lambda k: final_rep[k], reverse=True)

            top_items = []
            for item_index in final_rep_sorted:
                top_items.append(self.id_item_mapper[item_index])
            ret_dict[user] = top_items
        return ret_dict
