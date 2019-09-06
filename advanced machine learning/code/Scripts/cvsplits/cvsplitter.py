import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import pickle


def kFold(df, n_splits, test_size, random_state):
    """kFold

        kFold splitting.

        Arguments:
        ---------
        df: Pandas dataframe
            dataframe with the raw data
        no_splits: int
            number of different splittings of the data (k)
        test_size: float
            % of the data for the validation set
        random_state: int
            random state

        Output:
        ------
        indexes: np.array
            size: (n_splits (k), 2). each of the 2 are the indexes of the train and validation set, respectively
            for

        """

    folds = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state= random_state)
    folds_generator = folds.split(df, df.loc[:, ['Class']])
    return folds_generator


def test_splits(df, folds):

    for i, fold in enumerate(folds):
        print("Fold " + str(i))

        train_set = df.iloc[fold[0]]
        print("Train set: " + str(len(fold[0])) + ", with " + str(train_set['Class'].mean() * 100) + "% fraud cases.")
        print("First indexes: " + str(fold[0][1:5]))

        test_set = df.iloc[fold[1]]
        print("Test set: " + str(len(fold[1])) + ", with " + str(test_set['Class'].mean() * 100) + "% fraud cases.")
        print("First indexes: " + str(fold[1][1:5]))

        print("-------------------------------")


if __name__ == '__main__':

    SPLITS = 10
    TESTSIZE = 0.05
    RANDOMSTATE = 0

    df = pd.read_csv('../../Dataset/creditcard.csv')

    folds = kFold(df, n_splits=SPLITS, test_size=TESTSIZE, random_state=RANDOMSTATE)

    # Materialise values from generator
    folds = list(folds)

    with open('cvsplits_do_not_use.pkl', 'wb') as f:
        pickle.dump([x for x in folds], f)

    test_splits(df=df, folds=folds)
