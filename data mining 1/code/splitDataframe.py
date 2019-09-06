import numpy as np
import DBConnect as db
import pandas as pd
import os.path


def split_df(dataframe, split_percentage, group_on_order_ID):
    """
       ARGS
       master_FDM           a dataframe containing the imputed FDM data
       split_percentage     a float between [0,1]
       group_on_order_ID    boolean : True means data is grouped by order_id before splitting


       RETURNS
       a dictionary containing two dataframes called "model" and "threshold"
    """

# get the dataframe
    master_FDM = dataframe

    # get distinct order id list
    distint_order_id_set = master_FDM['order_id'].unique()
    np.random.seed(seed=1)
    msk = np.random.rand(len(distint_order_id_set)) < 1 - split_percentage
    model_order_ids = distint_order_id_set[msk]
    thresh_order_ids = distint_order_id_set[~msk]
    threshold = master_FDM.loc[master_FDM.order_id.isin(thresh_order_ids)]
    model = master_FDM.loc[master_FDM.order_id.isin(model_order_ids)]

    return [model, threshold]

if __name__ == "__main__":

    PERCENTAGE_SPLIT = 0.2

    # setting up the global path
    global_path = os.path.dirname(__file__)

    # 1. If the imputed_master_FDM is exsist then just load it from the feature data file
    if os.path.isfile(global_path + '/feature_data/data_train.pkl'):
        imputed_master_FDM = pd.read_pickle(global_path + '/feature_data/data_train.pkl')
    else:
        # connect to the database(call DBConnect)
        # replace the password and the database name to what is stored on your respective machines.
        connection = db.get_connection('root', '12345', 'instacart_features')

        # get the data(from the raw_Master FDM)
        # SQL query string
        """
        Note:   Use the below Query for testing as it took a long time to execute the code
                So just executed the query to extract all the features for user 1


                Once you guys are satisfied that the code is correct.
                Replace the sqlQuery with the following.

        sqlQuery = "select * from instacart_all_features.ALL_features"
        select * from instacart_features.ALL_features WHERE eval_set = 'train' AND order_id % 100 = 0
        """
        sqlQuery = "select * from instacart_features.ALL_features WHERE eval_set = 'train'"
        raw_master_FDM_cursor = db.execute_query(connection, sqlQuery)
        raw_master_FDM = pd.DataFrame(raw_master_FDM_cursor.fetchall())
        # 2. impute the data and set missing value as -1
        imputed_master_FDM = raw_master_FDM.fillna(value=-1)
        imputed_master_FDM.to_pickle(global_path + '/feature_data/data_train.pkl')

    # 2. split the data for training model and selecting threshold
    [FDM_model, FDM_threshold] = split_df(dataframe=imputed_master_FDM, split_percentage=PERCENTAGE_SPLIT, group_on_order_ID=True)
    FDM_model.to_pickle(global_path + '/feature_data/data_train_model.pkl')
    FDM_threshold.to_pickle(global_path + '/feature_data/data_train_threshold.pkl')