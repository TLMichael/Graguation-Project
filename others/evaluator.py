import torch
import torch.nn as nn
import numpy as np
from art.attacks import FastGradientMethod, ProjectedGradientDescent, CarliniLInfMethod, CarliniL2Method
from art.classifiers import PyTorchClassifier
from art.metrics import loss_sensitivity

from attack_utils import nb_classes, input_shape, attack_configs


class Evaluator(object):
    """Evaluator for adversarial robust"""
    def __init__(self, model, dataset):
        model.eval()
        self.model = model
        self.dataset = dataset
        optimizer = torch.optim.Adam(model.parameters())        # Useless
        self.nb_classes = nb_classes[dataset]
        self.classifier = PyTorchClassifier((0., 1.), model=self.model, loss=nn.CrossEntropyLoss(), optimizer=optimizer,
                                            input_shape=input_shape[dataset], nb_classes=self.nb_classes)
    
    def evaluate(self, data_loader):
        """Natural evaluation"""
        data_iter = iter(data_loader)
        examples, labels = next(data_iter)
        examples, labels = examples.cpu().numpy(), labels.cpu().numpy()
        preds = np.argmax(self.classifier.predict(examples), axis=1)
        acc = np.sum(preds == labels) / labels.shape[0]
        return acc

    def evaluate_fgsm(self, data_loader):
        """Adversarial evaluation by FGSM"""
        norm, eps = np.inf, attack_configs['FGSM'][self.dataset]['epsilon']
        adv_crafter = FastGradientMethod(self.classifier, norm=norm, eps=eps)

        data_iter = iter(data_loader)
        examples, labels = next(data_iter)
        examples, labels = examples.cpu().numpy(), labels.cpu().numpy()
        labels_one_hot = np.eye(self.nb_classes)[labels]
        examples_adv = adv_crafter.generate(examples, y=labels_one_hot)
        
        preds = np.argmax(self.classifier.predict(examples_adv), axis=1)
        acc = np.sum(preds == labels) / labels.shape[0]
        return acc
    
    def evaluate_pgd(self, data_loader, num_iter=40):
        """Adversarial evaluation by PGD"""
        norm, eps = np.inf, attack_configs['PGD'][self.dataset]['epsilon']
        eps_step = 2 * eps / num_iter
        adv_crafter = ProjectedGradientDescent(self.classifier, norm=norm, eps=eps,
                                               eps_step=eps_step, max_iter=num_iter, random_init=True)

        data_iter = iter(data_loader)
        examples, labels = next(data_iter)
        examples, labels = examples.cpu().numpy(), labels.cpu().numpy()
        labels_one_hot = np.eye(self.nb_classes)[labels]
        examples_adv = adv_crafter.generate(examples, y=labels_one_hot)

        preds = np.argmax(self.classifier.predict(examples_adv), axis=1)
        acc = np.sum(preds == labels) / labels.shape[0]
        return acc
    
    def evaluate_cw(self, data_loader):
        eps = attack_configs['PGD'][self.dataset]['epsilon']
        adv_crafter = CarliniLInfMethod(self.classifier, targeted=False, eps=eps)
        
        data_iter = iter(data_loader)
        examples, labels = next(data_iter)
        examples, labels = examples.cpu().numpy(), labels.cpu().numpy()
        labels_one_hot = np.eye(self.nb_classes)[labels]
        examples_adv = adv_crafter.generate(examples, y=labels_one_hot)

        preds = np.argmax(self.classifier.predict(examples_adv), axis=1)
        acc = np.sum(preds == labels) / labels.shape[0]
        return acc
    
    def evaluate_cw_l2(self, data_loader):
        adv_crafter = CarliniL2Method(self.classifier, targeted=False)
    
        data_iter = iter(data_loader)
        examples, labels = next(data_iter)
        examples, labels = examples.cpu().numpy(), labels.cpu().numpy()
        labels_one_hot = np.eye(self.nb_classes)[labels]
        examples_adv = adv_crafter.generate(examples, y=labels_one_hot)
    
        preds = np.argmax(self.classifier.predict(examples_adv), axis=1)
        acc = np.sum(preds == labels) / labels.shape[0]
        return acc

    def evaluate_robust(self, data_loader):
        data_iter = iter(data_loader)
        examples, labels = next(data_iter)
        examples, labels = examples.cpu().numpy(), labels.cpu().numpy()
        labels_one_hot = np.eye(self.nb_classes)[labels]
        
        losses = []
        # Compute loss with implicit batching
        batch_size = 256
        for batch_id in range(int(np.ceil(examples.shape[0] / float(batch_size)))):
            batch_index_1, batch_index_2 = batch_id * batch_size, (batch_id + 1) * batch_size
            batch = examples[batch_index_1:batch_index_2]
            batch_labels = labels_one_hot[batch_index_1:batch_index_2]

            loss = loss_sensitivity(self.classifier, batch, batch_labels)
            losses.append(loss * batch.shape[0])
        
        res = sum(losses) / examples.shape[0]
        return res
        


