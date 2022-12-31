from scipy.sparse import coo_matrix, csr_matrix
import caboose
import pandas as pd

class Pernir:
    def __init__(self,train_baskets,test_samples,user_index= 0):
        self.train_baskets = train_baskets
        self.test_samples = test_samples
        self.basket_items_dict = {}
        self.user_baskets_dict = {}
        self.user_sim_dict = {}
        self.user_neighbors = {}
        self.user_index = user_index
        self.batch_size = 10000

    def train(self):
        baskets_df = self.train_baskets[['basket_id', 'item_id', 'add_to_cart_order']].drop_duplicates()
        basket_items = baskets_df.sort_values(['basket_id', 'add_to_cart_order']).groupby(['basket_id'])['item_id'] \
            .apply(list).reset_index(name='items')
        self.basket_items_dict = dict(zip(basket_items['basket_id'],basket_items['items']))

        user_baskets_df = self.train_baskets[['basket_id','user_id']].drop_duplicates()
        user_baskets = user_baskets_df.groupby(['user_id'])['basket_id'].apply(list) \
            .reset_index(name='baskets')
        self.user_baskets_dict = dict(zip(user_baskets['user_id'],user_baskets['baskets']))

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
        item_dic = {}
        rev_item_dic = {}
        for i, item in enumerate(df_items):
            item_dic[item] = i
            rev_item_dic[i] = item
        user_dic = {}
        rev_user_dic = {}
        for i, user in enumerate(df_users):
            user_dic[user] = i
            rev_user_dic[i] = user

        df['uid'] = df['user'].apply(lambda x: user_dic[x])
        df['pid'] = df['item'].apply(lambda x: item_dic[x])

        n_users = len(set(df['user'].tolist()))
        n_items = len(set(df['item'].tolist()))

        userItem_mat = coo_matrix((df.score.values, (df.uid.values, df.pid.values)), shape=(n_users, n_items))
        representations = csr_matrix(userItem_mat)

        print('start of knn')
        similarities = caboose.Index(n_users, n_items, representations.indptr,
                                representations.indices, representations.data,
                                300)
        print('knn finished')

        for index, user in rev_user_dic.items():
            self.user_neighbors[user] = []
            for other_index, similarity in similarities.topk(index):
                self.user_neighbors[user].append(other_index)
                self.user_sim_dict[(user, rev_user_dic[other_index])] = similarity

    def user_predictions(self,user, input_items):
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

    def predict(self):
        test_inputs = self.test_samples['input_items'].apply(eval).tolist()
        test_users = self.test_samples['user_id'].tolist()

        predictions = []
        print("index:",self.user_index)
        for i, input_items in enumerate(test_inputs):
            if i% 1000 == 0:
                print(i)
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
            predicted_items = [x[0] for x in sorted_item_scores[:1000]]
            predictions.append(predicted_items)
        return predictions



