import pandas as pd

import sys
sys.path.insert(0, '../')
from datasplit.preprocessing import load_data


def get_pickle_cvs():
    """Outputs the train and test set of a single fold every time it is called."""

    splits = pd.read_pickle("../cvsplits/cvsplits.pkl")
    df = load_data('../../Dataset/creditcard.csv')

    for _, (training_ids, test_ids) in enumerate(splits):
        train = df.loc[training_ids]
        test = df.loc[test_ids]
        yield train, test


if __name__ == '__main__':

    for train, test in get_pickle_cvs():

        print("Train set: " + str(len(train)) + ", with " + str(train['Class'].mean() * 100) + "% fraud cases.")
        print("First indexes: " + str(train.head(2)))

        print("Test set: " + str(len(test)) + ", with " + str(test['Class'].mean() * 100) + "% fraud cases.")
        print("First indexes: " + str(test.head(2)))

        print("-------------------------------")
        break
