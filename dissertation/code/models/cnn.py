import torch
import torch.nn as nn

from torch.nn.init import kaiming_normal_, constant_
from models.mlp import MLP
from random import choice


class CNNEncoder(nn.Module):

    @staticmethod
    def get_output_len(sequence_len, padding, dilation, kernel_len, stride):
        return int((sequence_len + 2 * padding - dilation * (kernel_len - 1) - 1) / stride + 1)

    def __init__(self, sequence_len, num_convolutions, num_input_channels, num_feature_maps, kernel_len, stride,
                      use_batch_norm):

        super(CNNEncoder, self).__init__()

        batch_norm = [nn.BatchNorm1d(num_feature_maps)] if use_batch_norm else []
        padding = int((kernel_len - 1) / 2)
        dilation = 1
        next_sequence_len = sequence_len

        layers = []
        for i in range(num_convolutions):
            layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=num_input_channels if i == 0 else num_feature_maps,
                    out_channels=num_feature_maps,
                    kernel_size=kernel_len,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=1
                ),
                nn.ReLU(),
                *batch_norm
            ))
            next_sequence_len = CNNEncoder.get_output_len(next_sequence_len, padding, dilation, kernel_len, stride)

        self.sequence_len = next_sequence_len
        self.layers = nn.ModuleList(layers)
        self.initialise()

    def forward(self, x):
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x

    def initialise(self):
        def weights_init(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                constant_(m.bias, 0)

        self.apply(weights_init)


class CNN(nn.Module):

    @staticmethod
    def generate_params():
        params = dict()
        params['num_convolutions'] = choice([1, 2, 3, 4])
        params['num_feature_maps'] = choice([16, 32, 64, 128, 256])
        params['kernel_len'] = choice([3, 5, 7, 9, 11, 13, 15])
        params['stride'] = choice([1, 2, 3])
        params['use_batch_norm'] = choice([True, False])

        # params['num_convolutions'] = choice([4])
        # params['num_feature_maps'] = choice([64])
        # params['kernel_len'] = choice([15])
        # params['stride'] = choice([3])
        # params['use_batch_norm'] = choice([True])
        return params

    def __init__(self, num_input_channels, input_size, output_len, encoder_params, decoder_params):

        super(CNN, self).__init__()
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params

        self.encoder = CNNEncoder(
            sequence_len=input_size,
            num_input_channels=num_input_channels,
            **encoder_params
        )

        self.decoder = MLP(
            input_len=encoder_params['num_feature_maps'] * self.encoder.sequence_len,
            output_len=output_len,
            params=decoder_params,
            **decoder_params
        )

        self.initialise()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.decoder(x)

    def initialise(self):

        def weights_init(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                constant_(m.bias, 0)

        self.apply(weights_init)

    def get_params(self):
        return {'encoder': self.encoder_params, 'decoder': self.decoder_params}


# class DualCNN(nn.Module):
#     def __init__(self, num_input_channels, num_classes, num_feature_maps, num_hidden_units, dropout, kernel_size,
#                  act_fn):
#
#         super(DualCNN, self).__init__()
#
#         if act_fn == 'ReLU':
#             self.act_fn = nn.ReLU
#         elif act_fn == 'SELU':
#             self.act_fn = nn.SELU
#         else:
#             raise ValueError('"{}" is not a valid value for the activation function.'.format(act_fn))
#
#         stride_a = 2
#         padding_a = (kernel_size - 1) / 2
#         dilation_a = 1
#
#         self.conv1a = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=num_input_channels,
#                 out_channels=num_feature_maps,
#                 kernel_size=kernel_size,
#                 stride=stride_a,
#                 padding=padding_a,
#                 dilation=dilation_a,
#                 groups=1
#             ),
#             nn.BatchNorm1d(num_feature_maps),
#             self.act_fn(),
#         )
#
#         self.conv2a = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=num_feature_maps,
#                 out_channels=num_feature_maps,
#                 kernel_size=kernel_size,
#                 stride=stride_a,
#                 padding=padding_a,
#                 dilation=dilation_a,
#                 groups=1
#             ),
#             nn.BatchNorm1d(num_feature_maps),
#             self.act_fn(),
#         )
#
#         stride_b = 4
#         padding_b = (kernel_size - 1) / 2
#         dilation_b = 1
#
#         self.conv1b = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=num_input_channels,
#                 out_channels=num_feature_maps,
#                 kernel_size=kernel_size,
#                 stride=stride_b,
#                 padding=padding_b,
#                 dilation=dilation_b,
#                 groups=1
#             ),
#             nn.BatchNorm1d(num_feature_maps),
#             self.act_fn(),
#         )
#
#         self.conv2b = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=num_feature_maps,
#                 out_channels=num_feature_maps,
#                 kernel_size=kernel_size,
#                 stride=stride_b,
#                 padding=padding_b,
#                 dilation=dilation_b,
#                 groups=1
#             ),
#             nn.BatchNorm1d(num_feature_maps),
#             self.act_fn(),
#         )
#
#         self.l_out_1a = get_output_len(170, padding_a, dilation_a, kernel_size, stride_a)
#         self.l_out_2a = get_output_len(self.l_out_1a, padding_a, dilation_a, kernel_size, stride_a)
#         self.l_out_1b = get_output_len(170, padding_b, dilation_b, kernel_size, stride_b)
#         self.l_out_2b = get_output_len(self.l_out_1b, padding_b, dilation_b, kernel_size, stride_b)
#
#         self.linear1 = nn.Sequential(
#             nn.Linear(num_feature_maps * (self.l_out_2a + self.l_out_2b), num_hidden_units),
#             nn.Dropout(dropout),
#             self.act_fn()
#         )
#
#         self.output = nn.Linear(num_hidden_units, num_classes)
#         self.layers = nn.ModuleList([self.conv1a, self.conv2a, self.conv1b, self.conv2b, self.linear1, self.output])
#         self.initialise()
#
#     def forward(self, x):
#         x = x.transpose(1, 2)
#         xa = self.conv1a(x)
#         xa = self.conv2a(xa)
#         xb = self.conv1b(x)
#         xb = self.conv2b(xb)
#         x = torch.cat((xa, xb), dim=2)
#         x = x.view(x.size(0), -1)
#         x = self.linear1(x)
#         x = self.output(x)
#         return nn.functional.log_softmax(x, dim=1)
#
#     def initialise(self):
#         if self.act_fn == nn.ReLU:
#             a = 0
#         elif self.act_fn == nn.SELU:
#             a = 1
#
#         def weights_init(m):
#             if isinstance(m, (nn.Linear, nn.Conv1d)):
#                 kaiming_normal_(m.weight.data, a=a, mode='fan_in')
#                 constant_(m.bias, 0)
#
#         self.apply(weights_init)
