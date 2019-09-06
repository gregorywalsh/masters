import pandas as pd
import os.path
import re
import numpy as np

from scipy.stats import norm
from scipy.signal import resample, resample_poly
from extractors.torchdataset import PAMAP2Dataset
from sklearn.preprocessing import MinMaxScaler


class PAMAP2Extractor():

    #################################################################

    # The 54 column_names in the data files are organized as follows:
    # 1.	    timestamp (s)
    # 2.	    activity_ID (see below for the mapping to the activities)
    # 3.	    heart rate (bpm)
    # 4-20.	    IMU hand
    # 21-37.	IMU chest
    # 38-54.	IMU ankle
    #
    # The IMU sensory data contains the following column_names:
    # 1.	    temperature (C)
    # 2-4.	    3D-acceleration data (ms-2), scale: 16g, resolution: 13-bit
    # 5-7.	    3D-acceleration data (ms-2), scale: 6g, resolution: 13-bit
    # 8-10.	    3D-gyroscope data (rad/s)
    # 11-13.	3D-magnetometer data
    # 14-17.	orientation (invalid in this data collection)

    #################################################################

    def __init__(self, raw_data_folder, filenames, output_folder_path, required_sensor_regex_pattern, refresh_data):
        self.raw_data_folder = raw_data_folder
        self.filenames = filenames
        self.output_folder_path = output_folder_path
        self.required_sensor_regex_pattern = required_sensor_regex_pattern
        self.consolidated_filename = "consolidated.pk"
        self.corrected_filename = "corrected.pk"
        self.imputed_filename = "imputed.pk"
        self.enumerated_filename = "enumerated.pk"
        self.standardised_filename = "standardised.pk"
        self.df = None
        self.stage = None
        self.column_names = [
            'timestamp',
            'activity_id',
            'heart_rate',
            'hand_temp_1',
            'hand_acc16_1',
            'hand_acc16_2',
            'hand_acc16_3',
            'hand_acc6_1',
            'hand_acc6_2',
            'hand_acc6_3',
            'hand_gyro_1',
            'hand_gyro_2',
            'hand_gyro_3',
            'hand_mag_1',
            'hand_mag_2',
            'hand_mag_3',
            'hand_orient_1',
            'hand_orient_2',
            'hand_orient_3',
            'hand_orient_4',
            'chest_temp_1',
            'chest_acc16_1',
            'chest_acc16_2',
            'chest_acc16_3',
            'chest_acc6_1',
            'chest_acc6_2',
            'chest_acc6_3',
            'chest_gyro_1',
            'chest_gyro_2',
            'chest_gyro_3',
            'chest_mag_1',
            'chest_mag_2',
            'chest_mag_3',
            'chest_orient_1',
            'chest_orient_2',
            'chest_orient_3',
            'chest_orient_4',
            'ankle_temp_1',
            'ankle_acc16_1',
            'ankle_acc16_2',
            'ankle_acc16_3',
            'ankle_acc6_1',
            'ankle_acc6_2',
            'ankle_acc6_3',
            'ankle_gyro_1',
            'ankle_gyro_2',
            'ankle_gyro_3',
            'ankle_mag_1',
            'ankle_mag_2',
            'ankle_mag_3',
            'ankle_orient_1',
            'ankle_orient_2',
            'ankle_orient_3',
            'ankle_orient_4',
        ]

        pattern = re.compile(pattern=self.required_sensor_regex_pattern)
        self.sensor_columns = [sensor for sensor in self.column_names if re.search(pattern=pattern, string=sensor)]

        self.old_to_new_activity_id_map = {
            0: 12,
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            12: 7,
            13: 8,
            16: 9,
            17: 10,
            24: 11,
        }

        self.activity_id_to_name = {
            12: 'Other',
            0: 'Lying',
            1: 'Sitting',
            2: 'Standing',
            3: 'Walking',
            4: 'Running',
            5: 'Cycling',
            6: 'Nordic walking',
            7: 'Ascending stairs',
            8: 'Descending stairs',
            9: 'Vacuum cleaning',
            10: 'Ironing',
            11: 'Rope jumping'
        }

        self.filename_to_subject_id_map = {
            filename: int(filename[-5]) for filename in self.filenames
        }

        if refresh_data:
            for file_name in [self.consolidated_filename, self.corrected_filename, self.imputed_filename, self.enumerated_filename, self.standardised_filename]:
                if os.path.isfile(self.output_folder_path + file_name):
                    os.remove(self.output_folder_path + file_name)
            self.consolidate()
            self.correct()
            self.impute(method='akima')
            self.enumerate_activities()

    def consolidate(self, write_out=True):
        print('Consolidating Data')

        # Define dtypes
        float_pattern = re.compile(pattern='(acc|mag|orient|gyro|timestamp|heart)')
        df_dtypes = {
            column: 'float32' for column in self.column_names if re.search(pattern=float_pattern, string=column)
        }

        # Define required column_names
        required_column_names = ['activity_id'] + self.sensor_columns
        dataframes = []
        for filename in self.filenames:
            df_read = pd.read_csv(
                self.raw_data_folder + filename,
                sep=' ',
                header=None,
                names=self.column_names,
                usecols=required_column_names,
                dtype=df_dtypes,
                engine='c'
            )
            df_read['subject_id'] = self.filename_to_subject_id_map[filename]
            dataframes.append(df_read)

        self.df = pd.concat(dataframes)
        self.df.activity_id = self.df.activity_id.map(self.old_to_new_activity_id_map)
        self.df.reset_index(drop=True, inplace=True)

        self.stage = 'consolidated'
        if write_out:
            self.df.to_pickle(self.output_folder_path + self.consolidated_filename)


    def correct(self, write_out=True):
        print('Correcting Data')
        if self.stage != 'consolidated':
            if os.path.isfile(self.output_folder_path + self.consolidated_filename):
                self.df = pd.read_pickle(self.output_folder_path + self.consolidated_filename)
            else:
                self.consolidate()

        # Fix participant 108's left handedness
        problem_columns = [
            'hand_acc16_2',
            'hand_acc6_2',
            'hand_mag_2',
            'hand_gyro_3',
            'ankle_acc16_2',
            'ankle_acc6_2',
            'ankle_mag_2',
            'ankle_gyro_3'
        ]

        problem_columns_to_fix = list(set(problem_columns) & set(self.sensor_columns))
        self.df.loc[self.df['subject_id'] == 8, problem_columns_to_fix] *= -1

        self.stage = 'corrected'
        if write_out:
            self.df.to_pickle(self.output_folder_path + self.corrected_filename)


    def impute(self, method='linear', write_out=True):
        print('Imputing Missing Data')
        if self.stage != 'corrected':
            if os.path.isfile(self.output_folder_path + self.corrected_filename):
                self.df = pd.read_pickle(self.output_folder_path + self.corrected_filename)
            else:
                self.correct()

        # Interpolate missing data
        if method in ['linear', 'akima']:
            self.df[self.sensor_columns] = self.df.groupby('subject_id')[self.sensor_columns].apply(
                lambda group: group.interpolate(method=method)
            )

        elif method == 'drop':
            if 'heart_rate' in self.sensor_columns:
                self.df['heart_rate'] = self.df.groupby('subject_id')['heart_rate'].interpolate(method=method)
            self.df = self.df.loc[np.logical_not(self.df.isnull().values.any(axis=1))]

        if self.df.isnull().values.any():
            print("{} NaNs found in interpolated dataframe.".format(self.df.isnull().values.sum()))

        self.stage = 'imputed'
        if write_out:
            self.df.to_pickle(self.output_folder_path + self.imputed_filename)


    def enumerate_activities(self, write_out=True):

        print('Enumerating Activities')
        if self.stage != 'imputed':
            if os.path.isfile(self.output_folder_path + self.imputed_filename):
                self.df = pd.read_pickle(self.output_folder_path + self.imputed_filename)
            else:
                self.impute()

        # Create activity_instances
        self.df['activity_instance'] = 0
        last_subject_id = None
        last_activity_id = None
        instance = -1
        activity_instance = []
        for i, (activity_id, subject_id) in enumerate(zip(self.df.activity_id, self.df.subject_id)):
            if not (subject_id == last_subject_id and activity_id == last_activity_id):
                instance += 1
                last_subject_id = subject_id
                last_activity_id = activity_id
            activity_instance.append(instance)
            if i % 1000000 == 0:
                print(i, 'activities enumerated')
        self.df.activity_instance = activity_instance

        # Drop the 'other' activity now we have defined instances
        self.df = self.df.loc[self.df.activity_id != 12]

        self.stage = 'enumerated'
        if write_out:
            self.df.to_pickle(self.output_folder_path + self.enumerated_filename)

    def standardise(self, method='', write_out=True, test_subject_id=None, validation_subject_id=None):

        print('Standardising Activities')
        if self.stage != 'enumerated':
            if os.path.isfile(self.output_folder_path + self.enumerated_filename):
                self.df = pd.read_pickle(self.output_folder_path + self.enumerated_filename)
            else:
                self.enumerate_activities()

        if method == 'ranked':
            # Ranked value normalisation
            self.df[self.sensor_columns] = self.df.groupby('subject_id')[self.sensor_columns].rank(axis=0, method='dense')
            maxes = self.df[self.sensor_columns].groupby(self.df.subject_id).max(axis=0)
            self.df[self.sensor_columns] /= self.df.set_index('subject_id').join(maxes, how='right', lsuffix='_')[self.sensor_columns].values
            clip=0.00001
            self.df[self.sensor_columns] = self.df[self.sensor_columns].clip(lower=clip, upper=1-clip, axis=0)
            self.df[self.sensor_columns] = norm.ppf(self.df[self.sensor_columns].values)
        elif method == 'meanstd':
            means = self.df[self.sensor_columns].groupby(self.df.subject_id).mean()
            stds = self.df[self.sensor_columns].groupby(self.df.subject_id).std()
            self.df[self.sensor_columns] -= self.df.set_index('subject_id').join(means, how='right', lsuffix='_')[self.sensor_columns].values
            self.df[self.sensor_columns] /= self.df.set_index('subject_id').join(stds, how='right', lsuffix='_')[self.sensor_columns].values
        elif method == 'popmeanstd':
            rows_to_include = ~self.df['subject_id'].isin([test_subject_id, validation_subject_id])
            means = self.df.loc[rows_to_include, self.sensor_columns].mean()
            stds = self.df.loc[rows_to_include, self.sensor_columns].std()
            self.df[self.sensor_columns] -= means
            self.df[self.sensor_columns] /= stds
        # elif method == 'popminmax01':
        #     mins = self.df[self.sensor_columns].min()
        #     maxs = self.df[self.sensor_columns].max()
        #     self.df[self.sensor_columns] -= mins
        #     self.df[self.sensor_columns] /= maxs - mins
        elif method == 'popminmaxsymmetric':
            rows_to_include = ~self.df['subject_id'].isin([test_subject_id, validation_subject_id])
            mins = self.df.loc[rows_to_include, self.sensor_columns].min()
            maxs = self.df.loc[rows_to_include, self.sensor_columns].max()
            self.df[self.sensor_columns] -= mins
            self.df[self.sensor_columns] /= maxs - mins
            self.df[self.sensor_columns] *= 2
            self.df[self.sensor_columns] -= 1
        # elif method == 'ranked01':
        #     # Ranked value normalisation
        #     self.df[self.sensor_columns] = self.df.groupby('subject_id')[self.sensor_columns].rank(axis=0, method='dense')
        #     maxes = self.df[self.sensor_columns].groupby(self.df.subject_id).max(axis=0)
        #     self.df[self.sensor_columns] /= self.df.set_index('subject_id').join(maxes, how='right', lsuffix='_')[self.sensor_columns].values
        elif method == 'rankedsymmetric':
            # Ranked value normalisation
            self.df[self.sensor_columns] = self.df.groupby('subject_id')[self.sensor_columns].rank(axis=0, method='dense')
            maxes = self.df[self.sensor_columns].groupby(self.df.subject_id).max(axis=0)
            self.df[self.sensor_columns] /= self.df.set_index('subject_id').join(maxes, how='right', lsuffix='_')[self.sensor_columns].values
            self.df[self.sensor_columns] *= 2
            self.df[self.sensor_columns] -= 1
        else:
            raise AssertionError('"{}" is not a valid standardisation method'.format(method))

        self.stage = 'standardised'
        if write_out:
            self.df.to_pickle(self.output_folder_path + self.standardised_filename)


    def generate_datasets(self, device, test_subject_id, validation_subject_id, window_secs, downsampling_factor,
                          window_shift):

        print('Generating Datasets')
        if self.stage != 'standardised':
            if os.path.isfile(self.output_folder_path + self.standardised_filename):
                self.df = pd.read_pickle(self.output_folder_path + self.standardised_filename)
            else:
                self.standardise()

        df_test = self.df.loc[self.df['subject_id'] == test_subject_id]
        test_dataset = PAMAP2Dataset(
            device=device,
            df_in=df_test,
            sensor_columns=self.sensor_columns,
            window_secs=window_secs,
            downsampling_factor=downsampling_factor,
            stride=window_shift
        )

        df_train = self.df.loc[~self.df['subject_id'].isin([test_subject_id, validation_subject_id])]
        training_dataset = PAMAP2Dataset(
            device=device,
            df_in=df_train,
            sensor_columns=self.sensor_columns,
            window_secs=window_secs,
            downsampling_factor=downsampling_factor,
            stride=window_shift
        )

        df_val = self.df.loc[self.df['subject_id'] == validation_subject_id]
        validation_dataset = PAMAP2Dataset(
            device=device,
            df_in=df_val,
            sensor_columns=self.sensor_columns,
            window_secs=window_secs,
            downsampling_factor=downsampling_factor,
            stride=window_shift
        )

        return test_dataset, training_dataset, validation_dataset
