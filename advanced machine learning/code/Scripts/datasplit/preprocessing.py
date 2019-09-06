import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler
from math import pi, sin, cos, log


class CustomScaler(StandardScaler):
    """
    Subclass of sklearn.preprocessing.StandardScaler which allows for excluding certain columns.
    Can be included as part of an sklearn.pipeline.Pipeline object
    """

    def __init__(self, copy=True, with_mean=True, with_std=True, ignored_column_indexes=None, use_class_id=None):
        self.ignored_column_indexes = [] if ignored_column_indexes is None else ignored_column_indexes
        self.use_class_id=use_class_id
        super(CustomScaler, self).__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    def fit(self, X, y=None):
        X_fit = np.delete(X, self.ignored_column_indexes, axis=1)
        if self.use_class_id is not None:
            X_fit = X_fit[y == self.use_class_id]
        return super(CustomScaler, self).fit(X_fit, y)

    def transform(self, X, y='deprecated', copy=None):
        X_trans = np.delete(X, self.ignored_column_indexes, axis=1)
        return super(CustomScaler, self).transform(X=X_trans, y=y, copy=copy)


def get_processed_data(path='../../Dataset/creditcard.csv'):
    """
    Backward compatible method for creating a dataframe from the raw data

    :param path:    path of the file to load (default: '../../Dataset/creditcard.csv')
    :return:        a dataframe containg normalised credit card data, with radial time components
    """

    df = load_data(path=path)
    normalise(df_to_norm=df, df_reference=df)
    return df


def load_data(path, create_radial_time=True, create_categorical_time=False, keep_raw_time=False):
    """
    Used for loading data from the creditcard.csv data file. Some feature engineering. No normalisation.

    :param path:                        file path of credit card data
    :param create_radial_time:          include 2d radial time engineered features
    :param create_categorical_time:     include categorical time features such as part of day, hour, min, etc
    :param keep_raw_time:               include the raw time column in the output

    :return:                            a dataframe containing non-normalised credit card data plus engineered features
    """

    df = pd.read_csv(path)

    # Convert time to int seconds
    df['Time'] = df['Time'].astype(int)

    # Normalise amount - use log transform on distribution to normalise
    df['LogAmount'] = df['Amount'].apply(lambda x: 0.01 if x == 0 else log(x))
    df.drop('Amount', axis=1, inplace=True)  # drop untransformed 'Amount' column

    if create_radial_time:
        # Create cyclic time components
        df['RadianTime'] = df['Time'] * 2 * pi / 86400
        df['TimeComponent1'] = df['RadianTime'].apply(lambda x: sin(x))
        df['TimeComponent2'] = df['RadianTime'].apply(lambda x: cos(x))
        df.drop('RadianTime', axis=1, inplace=True)

    if create_categorical_time:
        df['Day'] = df['Time'].floordiv(86400)
        df['Hour'] = df['Time'].floordiv(3600) % 24
        df['Min'] = df['Time'].floordiv(60) % 60
        df['Second'] = df['Time'] % 60
        df['PartOfDay'] = df['Hour'].floordiv(3)

    if not keep_raw_time:
        df.drop('Time', axis=1, inplace=True)

    return df


def normalise(df_to_norm, df_reference=None, columns_not_to_norm=['Class'], normalise_radial_time_components=False):
    """
    Used to normalise a data frame. Requires a reference frame from which std and means of each variable will be
    calculated. Can be the same
    :param df_to_norm:                          the dataframe to which normalisation should be applied
    :param df_reference:                        an optional dataframe to used to generate the std's and means. If not
                                                provided
    :param columns_not_to_norm:                 a list of column names not to normalise (default: ['Class'])
    :param normalise_radial_time_components:    perform normalisation of radial time components (not advised)

    :return:                                    a normalised dataframe containing with the same structure as df_to_norm
    """

    if df_reference is not None:
        assert set(df_to_norm) == set(df_reference), 'df_to_norm and df_reference columns do not match'
    else:
        df_reference = df_to_norm

    if 'Class' not in columns_not_to_norm:
        warnings.warn('Class column not excluded. Class IDs will be normalised')

    if not normalise_radial_time_components:
        columns_not_to_norm += ['TimeComponent1', 'TimeComponent2']

    # Normalise appropriate columns
    norm_cols = [col for col in df_reference if col not in columns_not_to_norm]
    df_to_norm[norm_cols] = df_to_norm[norm_cols] - df_reference[norm_cols].mean()
    df_to_norm[norm_cols] = df_to_norm[norm_cols] / df_reference[norm_cols].std()


if __name__ == "__main__":


    # BASIC USAGE
    df = get_processed_data()

    # ADVANCED USAGE
    create_radial_time = True
    create_categorical_time = True

    df = load_data(path='../../Dataset/creditcard.csv',
                           create_radial_time=create_radial_time,
                           create_categorical_time=create_categorical_time)

    columns_not_to_norm = ['Class']

    if create_categorical_time:
        columns_not_to_norm += ['Day', 'Hour', 'Min', 'Second', 'PartOfDay']

    normalise(df_to_norm=df, df_reference=df, columns_not_to_norm=columns_not_to_norm)