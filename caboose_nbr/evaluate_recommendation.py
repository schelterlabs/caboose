from caboose_nbr.metrics import *

def evaluate(user_test_baskets_dict, user_predictions):
    ndcg_scores_dict = {}
    recall_scores_dict = {}
    for k in [10,20]:
        print(k)
        recall_scores = {}
        ndcg_scores = {}
        for user in user_test_baskets_dict:
            top_items = []
            if user in user_predictions:
                top_items = user_predictions[user]
            recall_scores[user] = recall_k(user_test_baskets_dict[user],top_items,k)
            ndcg_scores[user] = ndcg_k(user_test_baskets_dict[user],top_items,k)
        #print(zero)
        print('recall:',np.mean(list(recall_scores.values())))
        print('ndcg:',np.mean(list(ndcg_scores.values())))
        recall_scores_dict[k] = recall_scores
        ndcg_scores_dict[k] = ndcg_scores
        
    return recall_scores_dict,ndcg_scores_dict
