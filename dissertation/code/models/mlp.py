import torch.nn as nn

from torch.nn.init import kaiming_normal_, constant_
from random import choice


class MLP(nn.Module):

    @staticmethod
    def generate_params():
        params = dict()
        params['num_hidden_layers'] = choice([1, 2, 3])
        params['num_hidden_units'] = choice([64, 128, 256, 512, 1024])
        params['use_batch_norm'] = choice([True, False])
        params['dropout'] = choice([0, 0.25, 0.5, 0.75])
        return params

    def __init__(self, input_len, output_len, num_hidden_layers, num_hidden_units, dropout, use_batch_norm,
                 params):

        super(MLP, self).__init__()
        self.params = params

        batch_norm = [nn.BatchNorm1d(num_hidden_units)] if use_batch_norm else []
        layers = []
        for i in range(num_hidden_layers):
            layers.append(nn.Sequential(
                nn.Linear(in_features=input_len if i == 0 else num_hidden_units, out_features=num_hidden_units),
                nn.Dropout(dropout),
                nn.ReLU(),
                *batch_norm
            ))
        layers.append(nn.Linear(params['num_hidden_units'], output_len))
        self.layers = nn.ModuleList(layers)
        self.initialise()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return nn.functional.log_softmax(x, dim=1)

    def initialise(self):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                constant_(m.bias, 0)

        self.apply(weights_init)

    def get_params(self):
        return self.params
