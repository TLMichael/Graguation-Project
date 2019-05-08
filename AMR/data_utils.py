import torch.utils.data as data
from torchvision import datasets, transforms
from torch._six import int_classes as _int_classes
import numpy as np
import copy

from graph_constructor import GraphConstructor


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


class KNNSampler(data.Sampler):
    """Data sampler for k-Neighbors of each sample.
    
    Args:
        sampler (data.Sampler): Base sampler.
        anchor_size (int): Number of anchors.
        graph (GraphConstructor): K-NearestNeighbors graph by Euclidean distance
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``anchor_size * (knn + 1)``
    """
    def __init__(self, sampler, anchor_size, graph, drop_last=False):
        if not isinstance(sampler, data.Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(anchor_size, _int_classes) or anchor_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(anchor_size))
        if not isinstance(graph, GraphConstructor):
            raise ValueError("graph should be an instance of "
                             "graph_constructor.GraphConstructor, but got graph={}"
                             .format(graph))
        self.sampler = sampler
        self.anchor_size = anchor_size
        self.graph = graph
        self.knn = graph.knn
        self.drop_last = drop_last
    
    def __iter__(self):
        anchor = []
        for idx in self.sampler:
            anchor.append(idx)
            if len(anchor) == self.anchor_size:
                batch = copy.copy(anchor)
                indexes = self.graph.get_knn_index(np.asarray(anchor))
                batch += indexes.reshape(-1).tolist()
                yield batch
                anchor = []
        if len(anchor) > 0 and not self.drop_last:
            batch = copy.copy(anchor)
            indexes = self.graph.get_knn_index(np.asarray(anchor))
            batch += indexes.reshape(-1).tolist()
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.anchor_size
        else:
            return (len(self.sampler) + self.anchor_size - 1) // self.anchor_size


def test():
    data_dir = '../data/cifar10'
    dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms.ToTensor())
    newset = DatasetWrapper(dataset)
    print(newset[0])
    print('Done')
    

if __name__ == '__main__':
    test()


