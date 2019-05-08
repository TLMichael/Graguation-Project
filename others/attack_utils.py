import torch
import torch.nn as nn


def fgsm(model, x, y, epsilon=0.1, label_leaking=True):
    """Construct FGSM adversarial examples"""
    delta = torch.zeros_like(x, requires_grad=True)
    logits = model(x + delta)
    # Use the model's output instead of the true labels to avoid label leaking at training time.
    if not label_leaking:
        y = logits.max(dim=1)[1]
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def pgd_linf(model, x, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False, label_leaking=True):
    """Construct PGD adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(x, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(x, requires_grad=True)
    
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(x + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


attack_function = {
    'FGSM': fgsm,
    'PGD': pgd_linf,
}
attack_configs = {
    'FGSM': {
        'MNIST': {
            'epsilon': 0.1,
        },
        'SVHN': {
            'epsilon': 0.02,
        },
        'CIFAR10': {
            'epsilon': 4 / 255,
        },
        'CIFAR100': {
            'epsilon': 4 / 255,
        },
    },
    'PGD': {
        'MNIST': {
            'epsilon': 0.1,
            'alpha': 0.01,
            'num_iter': 20,
        },
        'SVHN': {
            'epsilon': 0.02,
            'alpha': 0.002,
            'num_iter': 20,
        },
        'CIFAR10': {
            'epsilon': 4 / 255,
            'alpha': 4 / 2550,
            'num_iter': 20,
        },
        'CIFAR100': {
            'epsilon': 4 / 255,
            'alpha': 4 / 2550,
            'num_iter': 20,
        },
    }
}

nb_classes = {
    'MNIST': 10,
    'SVHN': 10,
    'CIFAR10': 10,
    'CIFAR100': 100,
}

input_shape = {
    'MNIST': (1, 28, 28),
    'SVHN': (3, 32, 32),
    'CIFAR10': (3, 32, 32),
    'CIFAR100': (3, 32, 32),
}

