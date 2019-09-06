import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.init import kaiming_normal_, xavier_uniform_, constant_
from random import choice
from models.cnn import CNNEncoder


class RNN(nn.Module):

    @staticmethod
    def generate_params():
        params = {}
        params['unit_type'] = choice(['GRU'])
        params['num_hidden_units'] = choice([32, 64, 128, 256, 512])
        params['num_hidden_layers'] = choice([1, 2])
        params['dropout'] = choice([0, 0.25, 0.5, 0.75])
        return params

    def __init__(self, num_input_channels, output_len,
                 rnn_params):

        super(RNN, self).__init__()
        self.rnn_params = rnn_params

        if rnn_params['unit_type'] == 'LSTM':
            RNNUnit = nn.LSTM
        elif rnn_params['unit_type'] == 'GRU':
            RNNUnit = nn.GRU
        else:
            RNNUnit = None

        self.rnn = RNNUnit(
            input_size=num_input_channels,
            hidden_size=rnn_params['num_hidden_units'],
            num_layers=rnn_params['num_hidden_layers'],
            dropout=rnn_params['dropout'],
            batch_first=True,
            bidirectional=False
        )

        self.out = nn.Linear(rnn_params['num_hidden_units'], output_len)

        self.initialise()

    def forward(self, x, internal_state):
        x, internal_state = self.rnn(x, internal_state)
        x = x[:, -1, :]
        x = self.out(x)
        return F.log_softmax(x, dim=1), internal_state

    def initialise(self):

        def weights_init(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                kaiming_normal_(m.weight.data, mode='fan_in')
                constant_(m.bias, 0)
            if isinstance(m, (nn.GRU, nn.LSTM)):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_normal_(param)

        self.apply(weights_init)

    def get_params(self):
        return self.rnn_params


class CRNN(nn.Module):

    def __init__(self, num_input_channels, input_len, output_len, encoder_params, rnn_params):

        super(CRNN, self).__init__()

        self.encoder_params = encoder_params
        self.rnn_params = rnn_params

        self.encoder = CNNEncoder(
            sequence_len=input_len,
            num_input_channels=num_input_channels,
            **encoder_params
        )

        if rnn_params['unit_type'] == 'LSTM':
            RNNUnit = nn.LSTM
        elif rnn_params['unit_type'] == 'GRU':
            RNNUnit = nn.GRU
        else:
            RNNUnit = None

        self.rnn = RNNUnit(
            input_size=encoder_params['num_feature_maps'],
            hidden_size=rnn_params['num_hidden_units'],
            num_layers=rnn_params['num_hidden_layers'],
            dropout=rnn_params['dropout'],
            batch_first=True,
            bidirectional=False
        )

        self.out = nn.Linear(rnn_params['num_hidden_units'], output_len)

        self.initialise()


    def forward(self, x, internal_state):
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x, internal_state = self.rnn(x, internal_state)
        x = x[:, -1, :]
        x = self.out(x)
        return F.log_softmax(x, dim=1), internal_state


    def initialise(self):

        def weights_init(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                kaiming_normal_(m.weight.data, mode='fan_in')
                constant_(m.bias, 0)
            if isinstance(m, (nn.GRU, nn.LSTM)):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_normal_(param)

        self.apply(weights_init)

    def get_params(self):
        return {'encoder':self.encoder_params, 'decoder': self.rnn_params}


class BRNN(nn.Module):

    @staticmethod
    def generate_params():
        params = {}
        params['unit_type'] = choice(['GRU'])
        params['num_hidden_units'] = choice([64, 128, 256, 512])
        params['num_hidden_layers'] = choice([1, 2, 3])
        params['bidirectional'] = choice([True, False])
        params['dropout'] = choice([0, 0.25, 0.5, 0.75])
        return params

    def __init__(self, output_size, unit_type, num_input_channels, num_hidden_units, dropout, num_hidden_layers,
                 bidirectional, params_dict):

        super(BRNN, self).__init__()

        if unit_type == 'LSTM':
            self.rnn_unit = nn.LSTM
        elif unit_type == 'GRU':
            self.rnn_unit = nn.GRU
        self.num_feature_maps = num_input_channels
        self.num_hidden_units = num_hidden_units
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.bidirectional = bidirectional
        self.params_dict = params_dict

        self.rnn = self.rnn_unit(
            input_size=self.num_feature_maps,
            hidden_size=self.num_hidden_units,
            num_layers=self.num_hidden_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=True
        )
        self.sequencer = self.rnn_unit(
            input_size=self.num_feature_maps,
            hidden_size=self.num_hidden_units,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=True
        )
        self.linear = nn.Linear(self.num_hidden_units, output_size)
        self.initialise()

    def forward(self, x, internal_state):
        x, internal_state = self.rnn(x, internal_state)

        x = self.linear(x)
        return F.log_softmax(x, dim=1), internal_state

    def initialise(self):

        def weights_init(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                kaiming_normal_(m.weight.data, mode='fan_in')
                constant_(m.bias, 0)
            if isinstance(m, (nn.GRU, nn.LSTM)):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_normal_(param)

        self.apply(weights_init)

    def get_params(self):
        params = {}
        params['num_feature_maps'] = self.num_feature_maps
        params['num_hidden_units'] = self.num_hidden_units
        params['dropout'] = self.dropout
        params['bidirectional'] = self.bidirectional
        params['num_hidden_layers'] = self.num_hidden_layers
        return params


def get_l_out(l_in, padding, dilation, kernel_size, stride):
    return int( (l_in + 2 * padding - dilation * (kernel_size-1) -1) / stride + 1 )


