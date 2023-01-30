import numpy as np
from caboose_nbr.nbr_base import NBRBase
from scipy import sparse
import similaripy as sim
import pandas as pd

class UPCFr(NBRBase):
    def __init__(self, train_baskets, test_baskets,valid_baskets,basket_count_min=0,min_item_count =5 ,
                 r = 300, q = 7, alpha = 0.7,job_id = 10):
        super().__init__(train_baskets,test_baskets,valid_baskets,basket_count_min,min_item_count)
        self.r = r
        self.q = q
        self.alpha = alpha
        self.min_item_count = min_item_count
        self.user_item_scores = {}
        self.job_id = job_id

        data = self.train_baskets
        basket_per_user = data[['user_id','basket_id']].drop_duplicates().groupby('user_id') \
            .agg({'basket_id':'count'}).reset_index()
        valid_users = set(basket_per_user[basket_per_user['basket_id'] >= self.basket_count_min]['user_id'].tolist())

        item_per_basket = data[['item_id','basket_id']].drop_duplicates().groupby('item_id') \
            .agg({'basket_id':'count'}).reset_index()
        valid_items = set(item_per_basket[item_per_basket['basket_id'] >=  self.min_item_count]['item_id'].tolist())

        data = data[data['user_id'].isin(valid_users)]
        self.data = data[data['item_id'].isin(valid_items)]
        self.data["order_number"] = self.data.groupby("user_id")["date"].rank("dense", ascending=True)
        self.data['uid'] = self.data['user_id'].rank(method='dense')-1
        self.data['pid'] = self.data['item_id'].rank(method='dense')-1
        self.n_items = len(valid_items)

    def uwPopMat(self):
        '''
          Calculate the user popularity matrix with the given recency window
          In:
              df_train: train Dataframe
              n_items: #items
          Return :
              User-wise Popularity matrix in csr sparse format
        '''
        n_users = self.data.user_id.unique().shape[0]
        if (self.r>0):
            # Get the number of user baskets Bu
            BUCount = self.data.groupby(['uid'])['order_number'].max().reset_index(name='Bu')
            # Calculate the denominator which equal to Min(recency,Bu) for each user
            BUCount['denominator'] = np.minimum(BUCount['Bu'],self.r)
            # Calculater the order index, form where we start counting item appearance in recent orders
            BUCount['startindex'] = np.maximum(BUCount['Bu']-self.r,0)
            # Calcualte item appearance in recent orders
            tmp = pd.merge(BUCount, self.data,on='uid')[['uid','pid','order_number','startindex']]
            tmp = tmp.loc[(tmp['order']>=tmp['startindex'])==True].groupby(['uid','pid'])['order_number'].count().reset_index(name='numerator')
            tmp = pd.merge(BUCount[['uid','denominator']],tmp,on='uid')
            # finally calculate the recency aware user-wise popularity
            tmp['Score'] = tmp['numerator']/tmp['denominator']
        else :
            # Calculate user-wise popularity for each item
            BUCount = self.data.groupby(['uid'])['order_number'].max().reset_index(name='Bu')
            BUICount =  self.data.groupby(['uid','pid'])['basket_id'].count().reset_index(name='Bui')
            tmp = pd.merge(BUICount, BUCount, on='uid')
            del BUICount
            tmp['Score'] = tmp['Bui']/tmp['Bu']
            del BUCount
            # get the 3 columns needed to construct our user-wise Popularity matrix
        df_UWpop =  tmp[['uid','pid','Score']]
        del tmp
        # Generate user-wise popularity matrix in COOrdinate format
        print(df_UWpop.nunique())
        print(df_UWpop.shape)
        print(n_users,self.n_items)
        UWP_mat = sparse.coo_matrix((df_UWpop.Score.values, (df_UWpop.uid.values, df_UWpop.pid.values)), shape=(n_users,self.n_items))
        del df_UWpop
        return sparse.csr_matrix(UWP_mat)

    def train(self):
        UWP_sparse = self.uwPopMat()
        n_users = self.data['uid'].unique().shape[0]
        df_user_item = self.data.groupby(['uid','pid']).size().reset_index(name="bool")[['uid','pid']]
        # Generate the User_item matrix using the parse matrix COOrdinate format.
        userItem_mat = sparse.coo_matrix((np.ones((df_user_item.shape[0])), (df_user_item.uid.values, df_user_item.pid.values)), shape=(n_users,self.n_items))
        # Calculate the asymmetric similarity cosine matrix
        userSim = sim.cosine(sparse.csr_matrix(userItem_mat), k=1000)
        userSim.setdiag(0)
        # recommend k items to users
        self.user_recommendations = sim.dot_product(userSim.power(self.q), UWP_sparse, k=1000).toarray()

    def predict(self):
        ret_dict = {}

        user_id_map = self.data[['user_id','uid']].drop_duplicates()
        user_id_map_dict = dict(zip(user_id_map['user_id'],user_id_map['uid'].astype(int)))

        item_id_map = self.data[['item_id','pid']].drop_duplicates()
        item_id_map_dict = dict(zip(item_id_map['pid'].astype(int),item_id_map['item_id']))

        for user in user_id_map_dict:
            uid = user_id_map_dict[user]
            item_scores = self.user_recommendations[uid]
            top_indices =  item_scores.argsort()[-1000:][::-1]
            top_items = [item_id_map_dict[x] for x in top_indices]
            ret_dict[user] = top_items

        return ret_dict
