import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, init_size=7):
        super(UnFlatten, self).__init__()
        self.init_size = init_size
        
    def forward(self, input):
        return input.view(input.size(0), -1, self.init_size, self.init_size)


class AutoEncoder(nn.Module):
    def __init__(self, n_channels=1, img_size=28, z_dim=64):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, z_dim, kernel_size=3, stride=2, padding=1),
            Flatten(),
        )
        init_size = img_size // 4
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 32 * init_size * init_size),
            nn.ReLU(),
            UnFlatten(init_size),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, n_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x, only_logits=True):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        out = x_rec if only_logits else (x_rec, z)
        return out


def auto_encoder(dataset, z_dim=64):
    if dataset == 'MNIST':
        model = AutoEncoder(n_channels=1, img_size=28, z_dim=z_dim)
    elif dataset in ['CIFAR10', 'CIFAR100', 'SVHN']:
        model = AutoEncoder(n_channels=3, img_size=32, z_dim=z_dim)
    else:
        raise Exception('Invalid data set!', dataset)
    # print(model)
    return model


def test():
    import torch
    x = torch.rand(1, 1, 28, 28)
    model = AutoEncoder(n_channels=1, img_size=28, z_dim=64)
    # x = torch.rand(1, 3, 32, 32)
    # model = AutoEncoder(n_channels=3, img_size=32, z_dim=64)
    
    x_rec, z = model(x)
    assert x.shape == x_rec.shape
    print('Done')


if __name__ == '__main__':
    test()


