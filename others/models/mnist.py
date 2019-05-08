import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            current_dims = n_hidden

        self.features = nn.Sequential(layers)
        self.classifier = nn.Linear(current_dims, n_class)

    def forward(self, x, only_logits=True):
        x = x.view(x.size(0), -1)
        z = self.features(x)
        y = self.classifier(z)
        out = y if only_logits else (y, z)
        return out


def mnist(input_dims=784, n_hiddens=[256, 256, 64], n_class=10):
    model = MLP(input_dims, n_hiddens, n_class)
    # print(model)
    return model

