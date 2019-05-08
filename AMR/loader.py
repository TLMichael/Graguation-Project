import torch
from torch.utils import data
from torchvision import datasets, transforms
import numpy as np
import os

from data_utils import DatasetWrapper, KNNSampler


def get_loader(dataset_name, data_dir, batch_size, mode, shuffle, num_workers=1, graph=None, anchor_size=None):
    data_set = None
    data_dir = os.path.join(data_dir, dataset_name.lower())
    
    if dataset_name == 'MNIST':
        # Train/val/test: 45000/15000/10000
        n_train = 45000
        if mode == 'train':
            data_set = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
            data_set.train_data = data_set.train_data[:n_train]
            data_set.train_labels = data_set.train_labels[:n_train]
            data_set = DatasetWrapper(data_set)
        elif mode == 'val':
            data_set = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
            data_set.train_data = data_set.train_data[n_train:]
            data_set.train_labels = data_set.train_labels[n_train:]
        elif mode == 'test':
            data_set = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
    elif dataset_name == 'SVHN':
        # Train/val/test: 53257/20000/26032
        n_train = 53257
        if mode == 'train':
            data_set = datasets.SVHN(data_dir, split='train', download=True, transform=transforms.ToTensor())
            data_set.data = data_set.data[:n_train]
            data_set.labels = data_set.labels[:n_train]
            data_set = DatasetWrapper(data_set)
        elif mode == 'val':
            data_set = datasets.SVHN(data_dir, split='train', download=True, transform=transforms.ToTensor())
            data_set.data = data_set.data[n_train:]
            data_set.labels = data_set.labels[n_train:]
        elif mode == 'test':
            data_set = datasets.SVHN(data_dir, split='test', download=True, transform=transforms.ToTensor())
    elif dataset_name == 'CIFAR10':
        # Train/val/test: 45000/15000/10000
        n_train = 45000
        if mode == 'train':
            data_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.ToTensor())
            data_set.train_data = data_set.train_data[:n_train]
            data_set.train_labels = data_set.train_labels[:n_train]
            data_set = DatasetWrapper(data_set)
        elif mode == 'val':
            data_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.ToTensor())
            data_set.train_data = data_set.train_data[n_train:]
            data_set.train_labels = data_set.train_labels[n_train:]
        elif mode == 'test':
            data_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transforms.ToTensor())
    elif dataset_name == 'CIFAR100':
        # Train/val/test: 45000/15000/10000
        n_train = 45000
        if mode == 'train':
            data_set = datasets.CIFAR100(data_dir, train=True, download=True, transform=transforms.ToTensor())
            data_set.train_data = data_set.train_data[:n_train]
            data_set.train_labels = data_set.train_labels[:n_train]
            data_set = DatasetWrapper(data_set)
        elif mode == 'val':
            data_set = datasets.CIFAR100(data_dir, train=True, download=True, transform=transforms.ToTensor())
            data_set.train_data = data_set.train_data[n_train:]
            data_set.train_labels = data_set.train_labels[n_train:]
        elif mode == 'test':
            data_set = datasets.CIFAR100(data_dir, train=False, download=True, transform=transforms.ToTensor())
    else:
        raise Exception('Invalid data set!', dataset_name)
    
    if mode == 'train' and graph is not None:
        # For knn batch loader
        sampler = data.RandomSampler(data_set)
        knn_sampler = KNNSampler(sampler, anchor_size, graph)
        data_loader = data.DataLoader(data_set, batch_sampler=knn_sampler, num_workers=num_workers)
    else:
        batch_size = len(data_set) if (batch_size == np.inf) else batch_size
        data_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader


def get_model(dataset, model_save_dir=None):
    # lazy import
    from models.mnist import mnist
    from models.svhn import svhn
    from models.cifar import cifar10, cifar100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset == 'MNIST':
        model = mnist(input_dims=784, n_hiddens=[256, 256, 64], n_class=10)
    elif dataset == 'SVHN':
        model = svhn(n_channel=32)
    elif dataset == 'CIFAR10':
        model = cifar10(n_channel=128)
    elif dataset == 'CIFAR100':
        model = cifar100(n_channel=128)
    else:
        raise Exception('Invalid data set!', dataset)

    if model_save_dir is not None:
        # Load model checkpoints
        file_name = '{}.pt'.format(dataset.lower())
        file_path = os.path.join(model_save_dir, file_name)
        model.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
        print('Loaded model checkpoints from {}...'.format(file_path))

    return model.to(device)


def get_extractor_model(dataset, method, extractor_save_dir=None):
    if method == 'NT' or method == 'AT':
        model = get_model(dataset)
    elif method == 'AE':
        # Lazy import
        from models.ae import auto_encoder
        model = auto_encoder(dataset, z_dim=128)
    else:
        raise Exception('Invalid extractor!', method)
    
    if extractor_save_dir is not None:
        file_name = '{}-{}.pt'.format(dataset.lower(), method.lower())
        file_path = os.path.join(extractor_save_dir, file_name)
        model.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
        print('Loaded model checkpoints from {}...'.format(file_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device)


def test():
    dataset_name, data_dir, batch_size, mode, shuffle = 'SVHN', '../data', 128, 'train', True
    loader = get_loader(dataset_name, data_dir, batch_size, mode, shuffle)
    
    data_iter = iter(loader)
    item = next(data_iter)
    
    print('Max:', item[0].max())
    print('Min:', item[0].min())
    print('Done')


if __name__ == '__main__':
    test()

