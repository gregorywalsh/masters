import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os.path
from sklearn.externals import joblib

# setting up the global path
global_path = os.path.dirname(__file__)

# Load data_train_threshold
FDM_model = pd.read_pickle(global_path + '/feature_data/data_train_model.pkl')

# set which feature we want to use for training model
feature_col_names = ['aisle_DIST_CNT', 'days_between_orders_LOG_AVG',
                     'days_between_orders_LOG_STD', 'days_since_last_purchased_CNT',
                     'delay_between_purchases_LOG_AVG', 'delay_between_reorders_LOG_AVG',
                     'departments_DIST_CNT', 'occurances_per_order_PROP',
                     'orders_CNT', 'orders_TOT_COUNT', 'position_in_cart_LOG_AVG',
                     'position_in_cart_LOG_STD', 'prev_ord_as_pred_F1_AVG', 'prev_ord_as_pred_F1_STD',
                     'prev_ord_as_pred_PRECISION_AVG', 'prev_ord_as_pred_PRECISION_STD',
                     'prev_ord_as_pred_RECALL_AVG',
                     'prev_ord_as_pred_RECALL_STD', 'prods_ordered_DIST_CNT', 'prods_ordered_TOT_CNT',
                     'prods_per_order_LOG_AVG', 'prods_per_order_LOG_STD',
                     'proportion_of_orders_containing_PROP',
                     'proportion_reordered_PROP', 'purchased_CNT', 'reorders_CNT']


X = FDM_model[feature_col_names].as_matrix()

forest = joblib.load(global_path + '/model/randomforest.pkl')

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print(FDM_model[feature_col_names].columns[indices[f]])

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()