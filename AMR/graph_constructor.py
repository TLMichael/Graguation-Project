import torch
import os
from sklearn.neighbors import kneighbors_graph
import time
import datetime
import numpy as np
from scipy import sparse


class GraphConstructor(object):
    """
    K-NearestNeighbors graph by Euclidean distance.
    """
    def __init__(self, config):
        self.temperature = config.temperature
        self.knn = config.knn
        
        self.dataset = config.dataset
        self.method = config.extractor      # The choice of extractor
        assert self.method == 'NT' or self.method == 'AT' or self.method == 'AE'
        self.extractor_save_dir = config.extractor_save_dir
        
        self.graph_size = None      # Number of notes
        self.feature_name = '{}-{}-features.pt'.format(self.dataset.lower(), self.method.lower())
        self.feature_path = os.path.join(self.extractor_save_dir, self.feature_name)
        self.graph_name = '{}-{}-{}nn-graph.npz'.format(self.dataset.lower(), self.method.lower(), self.knn)
        self.graph_path = os.path.join(self.extractor_save_dir, self.graph_name)
        self.graph_connectivity_name = '{}-{}-{}nn-graph2.npz'.format(self.dataset.lower(), self.method.lower(), self.knn)
        self.graph_connectivity_path = os.path.join(self.extractor_save_dir, self.graph_connectivity_name)

        self.tensor = torch.load(self.feature_path)
        self.graph = None
        self.graph_connectivity = None
    
    def create_graph(self):
        if not os.path.exists(self.graph_path):
            print('Computing k-Neighbors graph...')
            X = self.tensor.cpu().numpy()
            start_time = time.time()
            self.graph = kneighbors_graph(X, self.knn, mode='distance', include_self=True, n_jobs=-1)
            self.graph_connectivity = kneighbors_graph(X, self.knn, mode='connectivity', include_self=True, n_jobs=-1)
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            print('Time cost for KNN graph: ', et)
    
            sparse.save_npz(self.graph_path, self.graph)
            sparse.save_npz(self.graph_connectivity_path, self.graph_connectivity)
            print('Saved KNN graph into {}...'.format(self.graph_path))
        print('Using computed k-Neighbors graph: {}'.format(self.graph_path))
        self.graph = sparse.load_npz(self.graph_path)
        self.graph_connectivity = sparse.load_npz(self.graph_connectivity_path)
    
    def get_knn_index(self, item):
        assert isinstance(item, int) or isinstance(item, np.ndarray)
        if self.graph is None:
            self.create_graph()
        knn = self.graph_connectivity[item]
        indexes = knn.indices
        if indexes.shape[0] == 31:
            print('fuck')
        if isinstance(item, np.ndarray):
            indexes = indexes.reshape(item.size, -1)
            indexes = np.fliplr(indexes).copy()    # Ensure order
        return indexes
    
    def get_knn_distance(self, item):
        assert isinstance(item, int) or isinstance(item, np.ndarray)
        if self.graph is None:
            self.create_graph()
        knn = self.graph_connectivity[item]
        indexes = knn.nonzero()
        distances = self.graph[item][indexes]
        distances = np.asarray(distances).squeeze()
        if isinstance(item, np.ndarray):
            distances = distances.reshape(item.size, -1)
            distances = np.fliplr(distances).copy()        # Ensure order
        return distances
    
    def get_similarity(self, indices, labels):
        """Similarity of batch examples"""
        # Unsupervised similarity matrix
        notes = self.tensor[indices]
        batch_size = notes.size(0)
        a = notes.unsqueeze(1).expand(batch_size, batch_size, -1)
        b = notes.unsqueeze(0).expand(batch_size, batch_size, -1)
        euclidean_distance = ((a - b)**2).sum(dim=2)
        similarity = torch.exp(-euclidean_distance / self.temperature)
        
        # Supervised similarity matrix
        labels = labels.to(notes.device)
        temp_a = labels.repeat(labels.shape[0], 1)
        temp_b = labels.unsqueeze(1).repeat(1, labels.shape[0])
        mask_intrinsic = (temp_a == temp_b).type(dtype=torch.float32)       # Intrinsic mask
        mask_penalty = (temp_a != temp_b).type(dtype=torch.float32)         # Penalty mask
        
        matrix_intrinsic = mask_intrinsic * similarity      # Intrinsic matrix
        matrix_penalty = mask_penalty * similarity          # Penalty matrix
        return matrix_intrinsic, matrix_penalty
        

def test():
    import argparse
    parser = argparse.ArgumentParser()
    config = parser.parse_args()
    config.temperature = 100
    config.knn = 16
    config.extractor = 'AE'
    config.dataset = 'MNIST'
    config.extractor_save_dir = './results/extractor'
    config.data_dir = '../data'
    if not os.path.exists(config.extractor_save_dir):
        os.makedirs(config.extractor_save_dir)
    graph = GraphConstructor(config)
    
    indexes = graph.get_knn_index(1)
    print(indexes)
    
    from loader import get_loader
    import numpy as np
    data_loader = get_loader(config.dataset, config.data_dir, batch_size=128, mode='train', shuffle=False)
    data_iter = iter(data_loader)
    idx, (x, y) = next(data_iter)
    intrinsic, penalty = graph.get_similarity(idx, y)
    print(intrinsic)
    print(penalty)


if __name__ == '__main__':
    test()


