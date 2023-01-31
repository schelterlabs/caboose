from scipy.sparse import coo_matrix, csr_matrix
import caboose
import pandas as pd
import similaripy 
from scipy.sparse.linalg import norm
import numpy as np

class Pernir:
    def __init__(self,train_baskets,test_samples,mode = 'similaripy',user_index= 0):
        self.train_baskets = train_baskets
        self.test_samples = test_samples
        self.basket_items_dict = {}
        self.user_baskets_dict = {}
        self.user_sim_dict = {}
        self.user_neighbors = {}
        self.user_index = user_index
        self.batch_size = 10000
        self.mode = mode
        
        all_items = set(self.train_baskets['item_id'].tolist())
        all_users = set(self.train_baskets['user_id'].tolist())
        self.item_dic = {}
        self.rev_item_dic = {}
        for i, item in enumerate(all_items):
            self.item_dic[item] = i
            self.rev_item_dic[i] = item
        self.user_dic = {}
        self.rev_user_dic = {}
        for i, user in enumerate(all_users):
            self.user_dic[user] = i
            self.rev_user_dic[i] = user
        
        self.compute_basket_dicts()
        
    def compute_basket_dicts(self):
        baskets_df = self.train_baskets[['basket_id', 'item_id', 'add_to_cart_order']].drop_duplicates()
        basket_items = baskets_df.sort_values(['basket_id', 'add_to_cart_order']).groupby(['basket_id'])['item_id'] \
            .apply(list).reset_index(name='items')
        self.basket_items_dict = dict(zip(basket_items['basket_id'],basket_items['items']))

        user_baskets_df = self.train_baskets[['basket_id','user_id']].drop_duplicates()
        user_baskets = user_baskets_df.groupby(['user_id'])['basket_id'].apply(list) \
            .reset_index(name='baskets')
        self.user_baskets_dict = dict(zip(user_baskets['user_id'],user_baskets['baskets']))
        
    def train(self):

        item_base_scores = {}
        for user in self.user_baskets_dict:
            baskets = self.user_baskets_dict[user]
            basket_len = len(baskets)
            if user not in item_base_scores:
                item_base_scores[user] = {}
                for basket_index, basket in enumerate(baskets):
                    w1_b = 1. / float(basket_len - basket_index)
                    for item in self.basket_items_dict[basket]:
                        if item not in item_base_scores[user]:
                            item_base_scores[user][item] = 0
                        item_base_scores[user][item] += w1_b

        data_list = {'user': [], 'item': [], 'score': []}

        for user in item_base_scores:
            baskets = self.user_baskets_dict[user]
            basket_len = len(baskets)
            for item in item_base_scores[user]:
                score = float(item_base_scores[user][item]) / float(basket_len)
                data_list['user'].append(user)
                data_list['item'].append(item)
                data_list['score'].append(score)

        df = pd.DataFrame.from_dict(data_list)

        df_users = set(df['user'].tolist())
        df_items = set(df['item'].tolist())


        df['uid'] = df['user'].apply(lambda x: self.user_dic[x])
        df['pid'] = df['item'].apply(lambda x: self.item_dic[x])

        n_users = len(self.user_dic)
        n_items = len(self.item_dic)

        userItem_mat = coo_matrix((df.score.values, (df.uid.values, df.pid.values)), shape=(n_users, n_items))
        representations = csr_matrix(userItem_mat)
        
        print('start of knn')
        if self.mode == 'caboose':
            self.caboose = caboose.Index(n_users, n_items, representations.indptr,
                                    representations.indices, representations.data,
                                    50)
            for index, user in self.rev_user_dic.items():
                self.user_neighbors[user] = []
                for other_index, similarity in self.caboose.topk(index):
                    self.user_neighbors[user].append(self.rev_user_dic[other_index])
                    self.user_sim_dict[(user, self.rev_user_dic[other_index])] = similarity
        elif self.mode == 'similaripy':
            self.user_neighbors = {}
            self.user_sim_dict = {}
            userSim = similaripy.cosine(csr_matrix(userItem_mat), k=50+1)
            this_user_sim_dict = dict(userSim.todok().items()) # convert to dictionary of keys format
            for key in this_user_sim_dict:
                self.user_sim_dict[(self.rev_user_dic[key[0]],self.rev_user_dic[key[1]])] = np.float32(this_user_sim_dict[key])
            for key in self.user_sim_dict:
                if key[0] not in self.user_neighbors:
                    self.user_neighbors[key[0]] = []
                if key[0] != key[1]:
                    self.user_neighbors[key[0]].append(key[1])
        print('knn finished')

    def forget_interactions(self, user_item_pairs):

        self.train_baskets['user_item'] = self.train_baskets.apply(lambda x: (x['user_id'], x['item_id']), axis=1)
        self.train_baskets = self.train_baskets[~self.train_baskets['user_item'].isin(user_item_pairs)]
        self.train_baskets.drop('user_item', axis=1, inplace=True)


        self.compute_basket_dicts()
        
        if self.mode == 'caboose':
            for user,item in user_item_pairs:
                if item in self.item_dic and user in self.user_dic:
                    self.caboose.forget(self.user_dic[user],self.item_dic[item])
            for index, user in self.rev_user_dic.items():
                self.user_neighbors[user] = []
                for other_index, similarity in self.caboose.topk(index):
                    self.user_neighbors[user].append(self.rev_user_dic[other_index])
                    self.user_sim_dict[(user, self.rev_user_dic[other_index])] = similarity
        if self.mode == 'similaripy':
            self.train()


    def user_predictions(self,user, input_items):

        if user not in self.user_baskets_dict:
            return {}

        baskets = self.user_baskets_dict[user]
        basket_len = len(baskets)

        item_base_scores = {}
        for basket_index,basket in enumerate(baskets):
            w1_b = 1./float(basket_len - basket_index)
            for item in self.basket_items_dict[basket]:
                if item not in item_base_scores:
                    item_base_scores[item] = 0
                item_base_scores[item] += w1_b

        current_scores = {}
        current_items_len = len(input_items)
        for current_item_index, current_item in enumerate(input_items):
            w2_j = 1./float(current_items_len - current_item_index)
            for basket_index,basket in enumerate(baskets):
                if current_item in self.basket_items_dict[basket]:
                    w1_b = 1./float(basket_len - basket_index)
                    i_index = self.basket_items_dict[basket].index(current_item)
                    for j_index,item in enumerate(self.basket_items_dict[basket]):
                        if i_index == j_index:
                            continue
                        w3_ij = 1./float(abs(i_index - j_index))
                        if item not in current_scores:
                            current_scores[item] = 0
                        current_scores[item] += w3_ij * w1_b * w2_j

        alpha1 = 0.3
        alpha2 = (1-alpha1)
        final_item_scores = {}
        for item in item_base_scores:
            final_item_scores[item] = alpha1 * item_base_scores[item]
            if item in current_scores:
                final_item_scores[item] += alpha2 * current_scores[item]

        return final_item_scores

    def predict_for_user(self, user, how_many):
        personal_scores = self.user_predictions(user, [])
        neighbor_scores = {}
        for neighbor in self.user_neighbors[user]:
            if neighbor == user:
                continue
            scores = self.user_predictions(neighbor, [])
            neighbor_scores[neighbor] = scores

        agg_neighbor_scores = {}
        norm_term = {}
        for neighbor in neighbor_scores:
            sim = self.user_sim_dict[(user, neighbor)]
            item_scores = neighbor_scores[neighbor]
            for item in item_scores:
                if item not in agg_neighbor_scores:
                    agg_neighbor_scores[item] = 0
                    norm_term[item] = 0
                agg_neighbor_scores[item] += item_scores[item] * sim
                norm_term[item] += sim

        beta1 = 0.3
        beta2 = (1-beta1)
        final_item_scores = {}
        for item in personal_scores:
            final_item_scores[item] = beta1 * personal_scores[item]

        for item in agg_neighbor_scores:
            if item not in final_item_scores:
                final_item_scores[item] = 0
            final_item_scores[item] += beta2 * (float(agg_neighbor_scores[item])/float(len(neighbor_scores)))

        sorted_item_scores = sorted(final_item_scores.items(),key= lambda x:x[1], reverse=True)
        predicted_items = [x[0] for x in sorted_item_scores[:how_many]]

        return predicted_items

    def predict(self):
        test_inputs = self.test_samples['input_items'].apply(eval).tolist()
        test_users = self.test_samples['user_id'].tolist()
    
        predictions = []
        prediction_scores = []
        print("num test users:",len(test_inputs))
        for i, input_items in enumerate(test_inputs):
            if i% 500 == 0:
                print(i, 'samples passed')
            user = test_users[i]
            input_items = test_inputs[i]
            current_items_len = len(input_items)

            
            
            personal_scores = self.user_predictions(user,input_items)
            neighbor_scores = {}
            for neighbor in self.user_neighbors[user]:
                if neighbor == user:
                    continue
                scores = self.user_predictions(neighbor,input_items)
                neighbor_scores[neighbor] = scores

            agg_neighbor_scores = {}
            norm_term = {}
            for neighbor in neighbor_scores:
                sim = self.user_sim_dict[(user,neighbor)]
                item_scores = neighbor_scores[neighbor]
                for item in item_scores:
                    if item not in agg_neighbor_scores:
                        agg_neighbor_scores[item] = 0
                        norm_term[item] = 0
                    agg_neighbor_scores[item] += item_scores[item]  * sim
                    norm_term[item] += sim

            beta1 = 0.3
            beta2 = (1-beta1)
            final_item_scores = {}
            for item in personal_scores:
                final_item_scores[item] = beta1 * personal_scores[item]

            for item in agg_neighbor_scores:
                if item not in final_item_scores:
                    final_item_scores[item] = 0
                final_item_scores[item] += beta2 * (float(agg_neighbor_scores[item])/float(len(neighbor_scores)))#/norm_term[item])

            sorted_item_scores = sorted(final_item_scores.items(),key= lambda x:x[1], reverse=True)
            predicted_items = [x[0] for x in sorted_item_scores[:50]]
            predicted_items_scores = [x[1] for x in sorted_item_scores[:50]]
            
            predictions.append(predicted_items)
            prediction_scores.append(predicted_items_scores)
        return predictions, prediction_scores
