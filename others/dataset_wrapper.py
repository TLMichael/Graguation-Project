import torch.utils.data as data
from torchvision import datasets, transforms


class DatasetWrapper(data.Dataset):
    """Dataset wrapper for adding index for each sample.
    
    Args:
        dataset (Dataset): The whole Dataset
    """
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index):
        return index, self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)


def test():
    data_dir = '../data/cifar10'
    dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.ToTensor())
    newset = DatasetWrapper(dataset)
    print(newset[0])
    print('Done')
    

if __name__ == '__main__':
    test()


