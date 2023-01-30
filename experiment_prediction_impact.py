import sys
import pandas as pd
import numpy as np
from caboose_nbr.tifuknn import TIFUKNN
from caboose_nbr.pernir import Pernir
from tqdm import tqdm

pd.options.mode.chained_assignment = None

def eprint(*args):
    print(*args, file=sys.stderr)


def evaluate_impact_on_instacart(model_to_test, description, sensitive_aisles, sample_size, seed):

    eprint(f'# Running: {model_to_test},{description},{seed},{sample_size}')

    np.random.seed(seed)

    train_baskets = pd.read_csv('data/instacart_30k/train_baskets.csv.gz')
    test_baskets = pd.read_csv('data/instacart_30k/test_baskets.csv')
    valid_baskets = pd.read_csv('data/instacart_30k/valid_baskets.csv')

    aisles = pd.read_csv('data/instacart_30k/aisles.csv')
    products = pd.read_csv('data/instacart_30k/products.csv')
    products_with_aisles = products.merge(aisles, on='aisle_id')

    train_baskets_with_aisles = train_baskets.merge(products_with_aisles,
                                                    left_on="item_id", right_on="product_id")

    baby_baskets = train_baskets_with_aisles[train_baskets_with_aisles.aisle_id.isin(sensitive_aisles)]
    all_baby_users = baby_baskets.user_id.unique()
    baby_users = np.array(np.random.choice(all_baby_users, sample_size))
    baby_user_baskets = train_baskets_with_aisles[train_baskets_with_aisles.user_id.isin(baby_users)]

    other_aisles = [aisle for aisle in baby_user_baskets.aisle_id.unique() if aisle not in sensitive_aisles]

    all_nonbaby_users = train_baskets_with_aisles[
        (train_baskets_with_aisles.aisle_id.isin(other_aisles)) \
        & (~train_baskets_with_aisles.user_id.isin(all_baby_users))].user_id.unique()

    nonbaby_users = np.array(np.random.choice(all_nonbaby_users, sample_size))

    #eprint(f'Sens: {len(baby_users)}, not sens: {len(nonbaby_users)}')

    users = np.concatenate((baby_users, nonbaby_users))
    sampled_train_baskets = train_baskets[train_baskets['user_id'].isin(users)]
    sampled_test_baskets = test_baskets[test_baskets['user_id'].isin(users)]
    sampled_valid_baskets = valid_baskets[valid_baskets['user_id'].isin(users)]

    if model_to_test == 'tifu':
        model = TIFUKNN(sampled_train_baskets, sampled_test_baskets, sampled_valid_baskets, 'caboose')
    elif model_to_test == 'pernir':
        model = Pernir(sampled_train_baskets, [], 'caboose')
    else:
        eprint("Unknown model...")
        sys.exit(-1)

    model.train()

    baby_items = set(baby_baskets.item_id.unique())

    num_affected_users = 0
    num_empty_afterwards = 0
    num_failing_users = 0

    for user in tqdm(baby_users):
        predictions = model.predict_for_user(user, 10)
        predicted_baby_items = set(predictions) & baby_items
        has_baby_items = len(predicted_baby_items) > 0
        if has_baby_items:
            chosen_users_items = sampled_train_baskets[sampled_train_baskets.user_id == user].item_id.unique()
            chosen_users_baby_items = set(chosen_users_items) & baby_items

            to_forget = [(user, item) for item in chosen_users_baby_items]
            model.forget_interactions(to_forget)
            predictions_after_forget = model.predict_for_user(user, 10)
            remaining_baby_items = set(predictions_after_forget) & baby_items

            num_affected_users += 1

            if len(remaining_baby_items) > 0:
                num_failing_users += 1

            if len(predictions_after_forget) == 0:
                num_empty_afterwards += 1

    print(f'PREDICTION_IMPACT,{model_to_test},{description},{seed},{sample_size},{num_affected_users},{num_failing_users},{num_empty_afterwards}')

#5,marinades meat preparation
#95,canned meat seafood
#96,lunch meat
#15,packaged seafood
#33,kosher foods
#34,frozen meat seafood
#35,poultry counter
#49,packaged poultry
#106,hot dogs bacon sausage
#122,meat counter

#27,beers coolers
#28,red wines
#62,white wines
#124,spirits
#134,specialty wines champagnes


baby_aisles = [82, 92, 102, 56]
meat_aisles = [5, 15, 33, 34, 35, 49, 95, 96, 106, 122]
alcohol_aisles = [27, 28, 62, 124, 134]
sample_size = 1000

for model_to_test in ['pernir', 'tifu']:
    for desc, sensitive in [('baby', baby_aisles), ('meat', meat_aisles), ('alcohol', alcohol_aisles)]:
        for seed in [43, 1312, 1234, 789, 1000, 7334, 7]:
            evaluate_impact_on_instacart(model_to_test, desc, sensitive, sample_size, seed)


