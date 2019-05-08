import os
import argparse
import numpy as np
from torch.backends import cudnn

from loader import get_loader, get_model
from feature_extractor import FeatureExtractor
from graph_constructor import GraphConstructor
from solvers import Solver_ATMR
from evaluator import Evaluator


def evaluate(config):
    # Data loader
    val_loader = get_loader(config.dataset, config.data_dir, batch_size=np.inf,
                            mode='val', shuffle=False, num_workers=config.num_workers)
    test_loader = get_loader(config.dataset, config.data_dir, batch_size=np.inf,
                             mode='test', shuffle=False, num_workers=config.num_workers)
    # Evaluating model
    model = get_model(config.dataset, model_save_dir=config.model_save_dir)
    evaluator = Evaluator(model, config.dataset)
    
    val_acc = {}
    print('Val set:')
    val_acc['cln'] = evaluator.evaluate(val_loader)
    print('Clean:', val_acc['cln'])
    val_acc['fgsm'] = evaluator.evaluate_fgsm(val_loader)
    print('FGSM:', val_acc['fgsm'])
    val_acc['pgd'] = evaluator.evaluate_pgd(val_loader, num_iter=50)
    print('PGD:', val_acc['pgd'])

    test_acc = {}
    print('Test set:')
    test_acc['cln'] = evaluator.evaluate(test_loader)
    print('Clean:', test_acc['cln'])
    test_acc['fgsm'] = evaluator.evaluate_fgsm(test_loader)
    print('FGSM:', test_acc['fgsm'])
    test_acc['pgd'] = evaluator.evaluate_pgd(test_loader, num_iter=50)
    print('PGD:', test_acc['pgd'])
    test_acc['cw'] = evaluator.evaluate_cw(test_loader)
    print('CW:', test_acc['cw'])
    test_acc['loss_sensitivity'] = evaluator.evaluate_robust(test_loader)
    print('loss_sensitivity:', test_acc['loss_sensitivity'])

    for i in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
        acc_pgd = evaluator.evaluate_pgd(test_loader, num_iter=i)
        print('PGD_{}: {}'.format(i, acc_pgd))
    
    return val_acc, test_acc


def main(config):
    # For fast training
    cudnn.benchmark = True
    
    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.extractor_save_dir):
        os.makedirs(config.extractor_save_dir)
    
    # Feature extractor
    extractor = FeatureExtractor(config)
    extractor.run()
    
    # Graph constructor
    graph = GraphConstructor(config)
    
    # Data loader
    train_loader = get_loader(config.dataset, config.data_dir, batch_size=config.batch_size,
                              mode='train', shuffle=True, num_workers=config.num_workers,
                              graph=graph, anchor_size=config.anchor_size)
    val_loader = get_loader(config.dataset, config.data_dir, batch_size=config.batch_size,
                            mode='val', shuffle=False, num_workers=config.num_workers)
    
    # Training model
    Solver = Solver_ATMR
    model = get_model(config.dataset)
    solver = Solver(train_loader, val_loader, graph, config)
    solver.train(model, config.attack_method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model configuration
    parser.add_argument('--lambda_item', type=float, default=0.1, help='The weight of manifold regularization')
    parser.add_argument('--temperature', type=float, default=10, help='The temperature of heat kernel')
    parser.add_argument('--knn', type=int, default=2, help='Number of k-Neighbors')
    parser.add_argument('--anchor_size', type=int, default=32, help='Number of anchors')
    parser.add_argument('--extractor', type=str, default='NT', choices=['NT', 'AT', 'AE'],
                        help='The choice of extractor: natural-trained classifier(NT),'
                             'adversarial-trained classifier(AT), auto-encoder(AE).')
    parser.add_argument('--attack_method', type=str, default='FGSM', choices=['FGSM', 'PGD'])
    # parser.add_argument('--mode', type=str, default='ATMR', choices=['NT', 'AT', 'ALP', 'ATDA', 'ATMR', 'ATDAMR'])
    
    # Training configuration
    parser.add_argument('--dataset', type=str, default='SVHN', choices=['MNIST', 'SVHN', 'CIFAR10', 'CIFAR100'])
    parser.add_argument('--batch_size', type=int, default=128, help='Mini-batch size')
    parser.add_argument('--test_batch_size', type=int, default=256, help='Mini-batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of total epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=30, help='Period of learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--extractor_epochs', type=int, default=10, help='Number of total epochs for training extractor')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)

    # Directories.
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--log_dir', type=str, default='./results/logs')
    parser.add_argument('--extractor_save_dir', type=str, default='./results/extractor')
    parser.add_argument('--model_save_dir', type=str, default='./results/checkpoints')

    config = parser.parse_args()
    import json

    print('Running parameters: ')
    print(json.dumps(vars(config), indent=4, separators=(',', ':')))
    
    # main(config)
    # evaluate(config)
    
    temperatures = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    temperatures = [10]
    knns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    knns = [5, 10, 50, 100]
    lambdas = [100, 10, 1, 0.1, 0.01, 0.001]
    lambdas = [0.1]
    vals = []
    tests = []
    for t in temperatures:
        for k in knns:
            for l in lambdas:
                config.temperature = t
                config.knn = k
                config.lambda_item = l
                main(config)
                val_acc, test_acc = evaluate(config)
                vals.append(val_acc)
                tests.append(test_acc)

    print(json.dumps(vals, indent=4))
    print(json.dumps(tests, indent=4))
    max_pgd = 0.
    max_idx = -1
    for idx, val in enumerate(vals):
        if val['pgd'] > max_pgd:
            max_pgd = val['pgd']
            max_idx = idx
    print('Best idx: ', max_idx)
    print('Best knn: ', knns[max_idx])
    print('Best test acc:', tests[max_idx])

