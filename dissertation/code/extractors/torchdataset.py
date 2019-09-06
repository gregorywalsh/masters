import torch
import numpy as np

from copy import copy

def downsample(arr, factor, method='first'):
    if method == 'mean':
        intermediate = np.cumsum(arr, 0)[factor - 1::factor] / factor
        intermediate[1:] = intermediate[1:] - intermediate[:-1]
    elif method == 'first':
        intermediate = arr[::factor]
    return intermediate


class TorchDataset():

    def __init__(self, inputs, targets, indexes=None, subjects=None):
        self.inputs = inputs
        self.targets = targets
        if indexes is not None:
            self.indexes = indexes
        if subjects is not None:
            self.subjects = subjects

    def get_class_weights(self):
        _, class_counts = np.unique(self.targets, return_counts=True)
        weights = np.divide(sum(class_counts) / len(class_counts), class_counts)
        return torch.from_numpy(weights).type(torch.FloatTensor)

    def get_sample_weights(self, class_weights):
        return class_weights[self.targets]

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        return torch.from_numpy(x).type(torch.FloatTensor), y

    def __len__(self):
        return self.inputs.shape[0]


class PAMAP2Dataset(TorchDataset):

    def __init__(self, df_in, device, sensor_columns, window_secs, downsampling_factor, stride):

        self.device = device

        # Trim and downsample
        downsampled_input_data = []
        targets = []
        subjects = []
        indexes = []
        offset = 0
        self.window_length = int(window_secs * 100 / downsampling_factor)
        for _, df_activity in df_in.groupby('activity_instance'):
            trimmed_input_data = df_activity[sensor_columns].iloc[1000:-1000].values
            downsampled_activity_data = downsample(trimmed_input_data, downsampling_factor, method='mean')
            num_datapoints = downsampled_activity_data.shape[0]
            if num_datapoints >= self.window_length:
                activity_id = df_activity['activity_id'].iloc[0]
                subject_id = df_activity['subject_id'].iloc[0]
                downsampled_input_data.append(downsampled_activity_data)
                indexes.append(np.arange(offset, offset + num_datapoints - self.window_length + 1, stride))
                targets.append(np.full(shape=indexes[-1].shape[0], fill_value=activity_id, dtype='int64'))
                subjects.append(np.full(shape=indexes[-1].shape[0], fill_value=subject_id, dtype='int64'))
                offset += num_datapoints

        self.indexes = np.concatenate(indexes, axis=0)
        self.inputs = torch.from_numpy(np.concatenate(downsampled_input_data, axis=0)).type(torch.FloatTensor).to(self.device)
        self.subjects = np.concatenate(subjects)
        self.targets = torch.from_numpy(np.concatenate(targets)).type(torch.LongTensor).to(self.device)

    def split(self, fraction=0.1):
        training_dataset_len = self.__len__()
        mask = np.full(training_dataset_len, False)
        mask[:int(training_dataset_len*fraction)] = True
        np.random.shuffle(mask)
        validation_dataset = copy(self)

        validation_dataset.indexes = self.indexes[mask]
        validation_dataset.subjects = self.subjects[mask]
        validation_dataset.targets = self.targets[mask]

        self.indexes = self.indexes[np.logical_not(mask)]
        self.subjects = self.subjects[np.logical_not(mask)]
        self.targets = self.targets[np.logical_not(mask)]

        return self, validation_dataset


    def __getitem__(self, index):
        x = self.inputs[self.indexes[index]:self.indexes[index] + self.window_length, :]
        y = self.targets[index]
        return x, y


    def __len__(self):
        return self.indexes.shape[0]