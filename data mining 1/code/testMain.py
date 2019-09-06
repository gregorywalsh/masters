import pandas as pd
import selectThreshold as st
import lightgbm as lgb
import os.path
from sklearn.externals import joblib

TRAIN_MODEL = "rf"
CSVFILENAME = "submission.csv"
PREDICTION_MODE = 'proba'

# setting up the global path
global_path = os.path.dirname(__file__)

# 1. Fetch the model that will be used and it's threshold
def get_model():
    if TRAIN_MODEL == 'LGB':
        # model = lgb.Booster(model_file=global_path + '/model/LGB_model.txt')

        model = joblib.load(global_path + '/model/lightGBM.pkl')
    else:
        model = joblib.load(global_path + '/model/randomforest.pkl')
    return model

def get_threshold():
    if PREDICTION_MODE == 'proba':
        threshold_FDM = pd.read_pickle(global_path + '/feature_data/data_train_threshold.pkl')
        threshold = st.get_best_threshold(get_model(), threshold_FDM, method=PREDICTION_MODE)
        return threshold[0]
    else:
        return 0.5

def to_kaggle(pred_orders_by_threshold):
    kaggle_orders = pred_orders_by_threshold[['order_id', 'product_id', 'order']].copy(deep=True)
    kaggle_orders.product_id = kaggle_orders.product_id.astype(str)
    kaggle_orders.loc[kaggle_orders.order == 0, 'product_id'] = None
    kaggle_orders = kaggle_orders.groupby('order_id')['product_id'].apply(lambda x: " ".join(filter(None, x)))
    return kaggle_orders

model = get_model()
best_threshold = get_threshold()

# 2. get test data pickle
test_FDM = pd.read_pickle(global_path+'/feature_data/data_test.pkl')

# 3. run the model on the test_fdm to get the product probabilities
print('predicting test data')
df_test_pred_prob = st.get_products_purchase_probabilities(model, test_FDM, mod=PREDICTION_MODE)

# 4. Generate predicted products by threshold
print('generating file')
test_pred_orders_by_threshold = st.generate_pred_order_by_threshold(best_threshold, df_test_pred_prob)

kaggle_orders = to_kaggle(test_pred_orders_by_threshold)
print(kaggle_orders.count())

# 5. Convert the dataframe of aggregated predicted test orders into a csv file for kaggle submission
kaggle_orders.to_csv(global_path + '/submissions/' + CSVFILENAME, header=["products"])
