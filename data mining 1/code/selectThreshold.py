'''
    #0 Import the Threshold FDM
    #1 Get the trained Model
    #2 Run the Model, which generates a dataframe of prediction probabilities
    #3 Append the correct target values from Threshold FDM
    #4 Generate candidate Thresholds from [0,1]
    #5 For each Threshold, generate the Set of Predicted orders
        #6 Compute F1 Score for each orders
        #7 Calcualate average F1 score
    #8 Select best Threshold
'''

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from sklearn.externals import joblib


def get_products_purchase_probabilities(trained_model, threshold_FDM, mod='binary'):
    # ARGS
    # trained_model  the trained Model
    # threshold_FDM  the FDM for thresholdFDM
    # mod 'binary' return 0 or 1 for prediction
    # mod 'proba' return a probability in range 0 to 1 for prediction
    #
    # Returns
    # df_pred_prob  the dataframe of prediction probabilities
    temp = threshold_FDM.copy(deep=True)
    df_user_ids = temp['user_id'].values
    df_order_ids = temp['order_id'].values
    df_product_ids = temp['product_id'].values
    target = temp['target'].values
    temp.drop('user_id', inplace=True, axis=1)
    temp.drop('order_id', inplace=True, axis=1)
    temp.drop('product_id', inplace=True, axis=1)
    temp.drop('eval_set', inplace=True, axis=1)
    temp.drop('target', inplace=True, axis=1)

    # pdb.set_trace()
    temp = temp.values
    if mod == 'binary':
        y = trained_model.predict(temp)
    elif mod == 'proba':
        y = trained_model.predict_proba(temp)
        y = np.array([result[1] for result in y])
    else :
        print('Please assign either "binary" or "proba" method for ' +
              'get_products_purchase_probabilities function.')

    df_pred_prob = pd.DataFrame({'user_id': df_user_ids,
                                 'order_id': df_order_ids,
                                 'product_id': df_product_ids,
                                 'prediction': y,
                                 'target': target}
                                )

    return df_pred_prob


def agg_target_orders(threshold_FDM):

    # ARGS
    # threshold_FDM  the FDM for thresholdFDM
    #
    # RETURNS
    # agg_target_orders  a groupBy object

    # Aggregates over the threshold_FDM to get the Target orders
    # 20180322 Ding:
    #   1. Due to the original name of variable is same as function name, added "ed" after agg instead.

    agged_target_orders = threshold_FDM[['user_id', 'order_id']].copy(deep=True)
    agged_target_orders = agged_target_orders.drop_duplicates()

    return agged_target_orders


def generate_thresholds():

    # ARGS
    # none
    #
    # Returns
    # thresholds a list of candidate thresholds

    # Using a search algoritm of ur choice, generate a list of threholds to test
    thresholds = np.linspace(0, 0.5, 100)

    return thresholds


def generate_pred_order_by_threshold(threshold, df_pred_prob):

    # ARGS
    # threshold  a single threshold
    # df_pred_prob
    #
    # RETURNS
    # agg_pred_orders_by_threshold

    # Based on the inputed threshold, aggregate on the dfPredProb to get the aggregated predicted orders for
    # that threshold

    agg_pred_orders_by_threshold = df_pred_prob.copy(deep=True)
    agg_pred_orders_by_threshold['order'] = np.where(agg_pred_orders_by_threshold['prediction'] >= threshold, 1, 0)

    return agg_pred_orders_by_threshold


def calc_avg_F1score(agg_pred_orders_by_threshold):

    # ARGS
    # agg_pred_orders_by_threshold
    # agg_target_orders
    #
    # RETURNS
    # avg_F1_score

    # Calcualates the avg F1 score for the inputted aggPredOrdersByThreshold against the
    agg_pred_orders_by_threshold['tp'] = np.where((agg_pred_orders_by_threshold['target'] == 1) &
                                                  (agg_pred_orders_by_threshold['order'] == 1), 1, 0)
    agg_pred_orders_by_threshold['fp'] = np.where((agg_pred_orders_by_threshold['target'] == 1) &
                                                  (agg_pred_orders_by_threshold['order'] == 0), 1, 0)
    agg_pred_orders_by_threshold['fn'] = np.where((agg_pred_orders_by_threshold['target'] == 0) &
                                                  (agg_pred_orders_by_threshold['order'] == 1), 1, 0)
    # agg_pred_orders_by_threshold['size'] = 1

    temp = agg_pred_orders_by_threshold.groupby(['user_id', 'order_id']).sum()

    f1s = (2 * temp['tp'] / (2 * temp['tp'] + temp['fp'] + temp['fn']))

    return np.mean(np.nan_to_num(f1s.values))


def get_best_threshold(trained_model, threshold_FDM, method='binary'):

    # ARGS
    # trained_model
    # threshold_FDM
    #
    # Returns
    # best_threshold
    # pdb.set_trace()
    # 1 Run the Model, which generates a dataframe of prediction probabilities
    df_pred_prob = get_products_purchase_probabilities(trained_model, threshold_FDM, method)

    # 4 Generate candidate Thresholds from [0,1]
    thresholds = generate_thresholds()
    avg_F1_scores = []

    f1max = 0
    best_threshold = 0

    # Loop over thresholds
    for t in thresholds:
        # 5 For each Threshold, generate the Set of Predicted orders
        agg_pred_orders_by_threshold = generate_pred_order_by_threshold(t, df_pred_prob)

        # 6 Calcualate average F1 score
        avg_F1_score = calc_avg_F1score(agg_pred_orders_by_threshold)

        avg_F1_scores.append(avg_F1_score)

        # 7 Select best Threshold
        if avg_F1_score > f1max:
            f1max = avg_F1_score
            best_threshold = t

        print(avg_F1_score)

    return best_threshold, avg_F1_scores, thresholds


def figure_plot(x, y, best_thr, path):
    # matplotlib.font_manager._rebuild()
    matplotlib.rcParams['font.sans-serif'] = ['Linux Biolinum', 'Tahoma', 'DejaVu Sans',
                                              'Lucida Grande', 'Verdana']
    # Here we initiate a single column figure
    fig = plt.figure(figsize=(5.33, 4), dpi=220)

    # Here's some code that makes some data to plot
    plt.plot(x, y)

    # Add the line for best threshold.
    # plt.axvline(x=best_thr)
    plt.plot(best_thr[0], best_thr[1], 'ro', markersize=4)
    plt.text(best_thr[0] + 0.005, best_thr[1], 'Best Threshold:' + "%0.4f" % (best_thr[0],), fontsize=8)

    # plt.xlabel('Value of Threshold')
    # plt.ylabel('F1 Score')
    plt.title('The F1 score of LightGBM model via different thresholds')

    # Use these parameters for consistency!
    # plt.legend(frameon=False) #You might have to add some extra arguments here to customise
    plt.tick_params(labelsize=8)
    plt.xlabel('Thresholds', fontweight='bold', fontsize=9)
    plt.ylabel('F1 Scores', fontweight='bold', fontsize=9)
    plt.grid(False)

    # This shows the figure, it may look slightly different from the PDF version
    # The the PDF version will normally be more nicely formatted
    plt.show()

    # This saves the figure to PDF with perfect clarity in the same folder as the script
    fig.savefig(fname=os.path.join(path, r'lgbm_threshold_F1score.pdf'), frameon=None,
                bbox_inches='tight', pad_inches=0, dpi='figure')
    # You may need to trim off the white space at the top of the PDF, this is easily done on Mac
    # But not sure on linux/windows

if __name__ == "__main__":
    # Locations of different files need to modify in different computers.
    features_path = r'D:\DataSet\DM_project\Features'
    fig_path = r'D:\Lei\GoogleDrive\DataScience\Sem2\COMP6237_DataMining\CW1\Group_git\data_mining'
    global_path = os.path.dirname(__file__)

    mod = 'lgbm'

    # Load feature data
    FDM_threshold = pd.read_pickle(os.path.join(features_path, 'FDM_threshold.pkl'))

    if mod == 'lgbm':
        # setting up the global path
        trained_model = joblib.load(os.path.join(global_path, r'model\lightGBM.pkl'))
        # trained_model = joblib.load(os.path.join(global_path, r'model\xgboost.pkl'))
        # FDM_threshold = pd.read_pickle(os.path.join(features_path, 'FDM_threshold.pkl'))

        best_threshold, F1_scores, thresholds = get_best_threshold(trained_model=trained_model,
                                                                   threshold_FDM=FDM_threshold, method='proba')

        # Save the value of best threshold
        f = open(r".\model\best_threshold.txt", "w+")
        f.write(str(best_threshold))      # str() converts to string
        f.close()

        print(best_threshold)

        # Rebuild the cache of matplotlib, need to import something from pylab.
        # Make sure the Linux Biolinum font installed before running this script.
        # Linux Biolinum: https://www.dafont.com/linux-biolinum.font
        # from pylab import *
        # matplotlib.font_manager._rebuild()

        figure_plot(thresholds, F1_scores, [best_threshold, max(F1_scores)], fig_path)
    elif mod == 'rf':
        # Random Forest
        # pdb.set_trace()
        trained_model = joblib.load(os.path.join(global_path, r'model\randomforest.pkl'))

        best_threshold, F1_scores, thresholds = get_best_threshold(trained_model=trained_model,
                                                                   threshold_FDM=FDM_threshold, method='binary')

        print(F1_scores)
