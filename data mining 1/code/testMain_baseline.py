import pandas as pd
import selectThreshold as st
import numpy as np
import os.path
from math import exp

TRAIN_MODEL = "LGB"

# setting up the global path
global_path = os.path.dirname(__file__)

def to_kaggle(pred_orders_by_threshold):
    kaggle_orders = pred_orders_by_threshold[['order_id', 'product_id', 'order']].copy(deep=True)
    kaggle_orders.product_id = kaggle_orders.product_id.astype(str)+" "
    kaggle_orders.loc[kaggle_orders.order == 0, 'product_id'] = ""
    kaggle_orders = kaggle_orders.groupby('order_id')['product_id'].apply(lambda x: "%s" % "".join(filter(None,x)))
    return kaggle_orders

def get_products_purchase_probabilities(df, predict_type='random'):
    # ARGS
    # trained_model  the trained Model
    # threshold_FDM  the FDM for thresholdFDM
    #
    #
    # Returns
    # df_pred_prob  the dataframe of prediction probabilities

    df_pred_prob = df[['user_id', 'order_id', 'product_id', 'target']]

    if predict_type == 'random':
        df_pred_prob['prediction'] = (np.random.rand(len(df)) < ((df['prods_ordered_TOT_CNT']/df['orders_TOT_COUNT'])/df['prods_ordered_DIST_CNT']))
    elif predict_type == 'historical':
        df_pred_prob['prediction'] = 1 * (df['proportion_of_orders_containing_PROP'] > 0.5)
    return df_pred_prob



CSVFILENAME = global_path+ "/submissions/submission_rand.5.csv"

# get test data pickle
test_FDM = pd.read_pickle(global_path+'/feature_data/data_test.pkl')

# 3. run the model on the test_fdm to get the predi
df_test_pred_prob = get_products_purchase_probabilities(df=test_FDM, predict_type='random')

# 5. Generate predicted products by threshold

test_pred_orders_by_threshold = st.generate_pred_order_by_threshold(0.5, df_test_pred_prob)

print(test_pred_orders_by_threshold.iloc[0])

kaggle_orders = to_kaggle(test_pred_orders_by_threshold)
print(kaggle_orders.count())

# 7. Convert the dataframe of aggregated predicted test orders into a csv file for kaggle submission
kaggle_orders.to_csv(CSVFILENAME, header=["products"])
print("end")