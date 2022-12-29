import pandas as pd
from scipy import sparse
import similaripy as sim


class PernirNonOptimized:

    _BASKET_ID_COLUMN = 'basket_id'
    _USER_ID_COLUMN = 'user_id'
    _ITEM_ID_COLUMN = 'item_id'
    _TIMESTAMP_COLUMN = 'add_to_cart_order'

    def __init__(self, k, alpha, beta):
        self.k = k
        self.alpha = alpha
        self.beta = beta

        self.is_trained = False
        # Index from users to baskets
        self.user_baskets_dict = {}
        # Index from baskets to items
        self.basket_items_dict = {}
        # User similarities
        self.user_neighbors = {}

    def train(self, train_baskets):

        self.user_neighbors = self._init_user_neighbors(train_baskets, self.k)

        user_baskets_dict, basket_items_dict = self._init_indexes(train_baskets)
        self.user_baskets_dict = user_baskets_dict
        self.basket_items_dict = basket_items_dict

        self.is_trained = True


    def predict_single(self, user, input_items):

        personal_scores = self._user_predictions(user, input_items)

        neighbor_scores = {}
        for neighbor in self.user_neighbors[user]:
            if neighbor == user:
                continue
            scores = self._user_predictions(neighbor, input_items)
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

        beta1 = self.beta
        beta2 = (1 - beta1)
        final_item_scores = {}
        for item in personal_scores:
            final_item_scores[item] = beta1 * personal_scores[item]

        for item in agg_neighbor_scores:
            if item not in final_item_scores:
                final_item_scores[item] = 0
            final_item_scores[item] += beta2 * (float(agg_neighbor_scores[item]) / float(len(neighbor_scores)))

        return final_item_scores


    def _user_predictions(self, user, input_items):
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

        alpha1 = self.alpha
        alpha2 = (1-alpha1)
        final_item_scores = {}
        for item in item_base_scores:
            final_item_scores[item] = alpha1 * item_base_scores[item]
            if item in current_scores:
                final_item_scores[item] += alpha2 * current_scores[item]

        return final_item_scores


    def _init_indexes(self, train_baskets):

        user_baskets_df = train_baskets[[self._BASKET_ID_COLUMN, self._USER_ID_COLUMN]] \
            .drop_duplicates()
        user_baskets = user_baskets_df \
            .groupby([self._USER_ID_COLUMN])[self._BASKET_ID_COLUMN] \
            .apply(list).reset_index(name='baskets')

        user_baskets_dict = dict(zip(user_baskets[self._USER_ID_COLUMN], user_baskets['baskets']))

        # TODO this is duplicated in the similarity computation
        baskets_df = train_baskets[[self._BASKET_ID_COLUMN, self._ITEM_ID_COLUMN, self._TIMESTAMP_COLUMN]] \
            .drop_duplicates()
        basket_items = baskets_df.sort_values([self._BASKET_ID_COLUMN, self._TIMESTAMP_COLUMN]) \
            .groupby([self._BASKET_ID_COLUMN])[self._ITEM_ID_COLUMN] \
            .apply(list).reset_index(name='items')
        basket_items_dict = dict(zip(basket_items[self._BASKET_ID_COLUMN], basket_items['items']))

        return user_baskets_dict, basket_items_dict



    def _init_user_neighbors(self, train_baskets, k):
        # TODO transactiondataframe might already do this
        baskets_df = train_baskets[[self._BASKET_ID_COLUMN, self._ITEM_ID_COLUMN, self._TIMESTAMP_COLUMN]] \
            .drop_duplicates()
        basket_items = baskets_df \
            .sort_values([self._BASKET_ID_COLUMN, self._TIMESTAMP_COLUMN]) \
            .groupby([self._BASKET_ID_COLUMN])[self._ITEM_ID_COLUMN] \
            .apply(list).reset_index(name='items')

        basket_items_dict = dict(zip(basket_items[self._BASKET_ID_COLUMN], basket_items['items']))

        user_baskets_df = train_baskets[[self._BASKET_ID_COLUMN, self._USER_ID_COLUMN]].drop_duplicates()
        user_baskets = user_baskets_df \
            .groupby([self._USER_ID_COLUMN])[self._BASKET_ID_COLUMN] \
            .apply(list).reset_index(name='baskets')

        user_baskets_dict = dict(zip(user_baskets[self._USER_ID_COLUMN], user_baskets['baskets']))

        item_base_scores = {}
        for user in user_baskets_dict:
            baskets = user_baskets_dict[user]
            basket_len = len(baskets)
            if user not in item_base_scores:
                item_base_scores[user] = {}
                for basket_index, basket in enumerate(baskets):
                    w1_b = 1. / float(basket_len - basket_index)
                    for item in basket_items_dict[basket]:
                        if item not in item_base_scores[user]:
                            item_base_scores[user][item] = 0
                        item_base_scores[user][item] += w1_b

        data_list = {'user': [], 'item': [], 'score': []}

        for user in item_base_scores:
            baskets = user_baskets_dict[user]
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
        userItem_mat = sparse.coo_matrix((df.score.values, (df.uid.values, df.pid.values)), shape=(n_users, n_items))

        sparse.save_npz('pernir-instacart.npz', sparse.csr_matrix(userItem_mat))
        print("Done")

        userSim = sim.asymmetric_cosine(sparse.csr_matrix(userItem_mat), alpha=0.5, k=k)
        user_sim_dict = dict(userSim.todok().items())  # convert to dictionary of keys format
        final_user_sim_dict = {}
        for key in user_sim_dict:
            final_user_sim_dict[(rev_user_dic[key[0]], rev_user_dic[key[1]])] = user_sim_dict[key]

        user_neighbors = {}
        for key in final_user_sim_dict:
            if key[0] not in user_neighbors:
                user_neighbors[key[0]] = []
            user_neighbors[key[0]].append(key[1])

        return user_neighbors