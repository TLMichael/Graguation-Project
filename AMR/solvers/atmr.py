import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
import datetime
from tensorboardX import SummaryWriter

from attack_utils import attack_function, attack_configs


class Solver_ATMR(object):
    """Solver for adversarial training with manifold training"""
    def __init__(self, train_loader, val_loader, graph, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.graph = graph
    
        # Training configuration
        self.dataset = config.dataset
        self.anchor_size = config.anchor_size
        self.knn = config.knn
        self.temperature = config.temperature
        self.epochs = config.epochs
        self.lr = config.lr
        self.step_size = config.step_size
        self.gamma = config.gamma
        self.lambda_item = config.lambda_item
        self.anchor_number = None

        # Directories
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir

        # Miscellaneous
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.dataset.lower()))
        self.postfix_tag = '{}/{}_{}'.format(self.dataset.lower(), self.temperature, self.knn)
        self.loss_tag = 'LOSS/' + self.postfix_tag
        self.accuracy_tag = 'ACC/' + self.postfix_tag
        self.similarity_tag = 'SIM/' + self.postfix_tag
        self.default_name = '{}.pt'.format(self.dataset.lower())
        self.default_path = os.path.join(self.model_save_dir, self.default_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, model, method='FGSM'):
        """Adversarial training with manifold regularization"""
        attack = attack_function[method]
        kwargs = attack_configs[method][self.dataset]
    
        opt = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=self.step_size, gamma=self.gamma)
        print('Start training...')
        start_time = time.time()
        best_acc_adv = 0.
        for epoch_idx in range(self.epochs):
            scheduler.step()
            train_loss, train_acc_cln, train_acc_adv = self.epoch_train(model, opt, attack, **kwargs)
            val_loss, val_acc_cln, val_acc_adv = self.epoch_val(model, attack, **kwargs)
        
            # Print out training information
            loss_logs = {
                'Train': train_loss,
                'Val': val_loss,
            }
            acc_logs = {
                'Train_cln': train_acc_cln,
                'Train_adv': train_acc_adv,
                'Val_cln': val_acc_cln,
                'Val_adv': val_acc_adv,
            }
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Epoch [{}/{}]".format(et, epoch_idx + 1, self.epochs)
            for tag, value in acc_logs.items():
                log += ", Acc/{}: {:.4f}".format(tag, value)
            print(log)
            self.writer.add_scalars(self.loss_tag, loss_logs, epoch_idx)
            self.writer.add_scalars(self.accuracy_tag, acc_logs, epoch_idx)
        
            # Save model checkpoints
            if val_acc_adv > best_acc_adv:
                best_acc_adv = val_acc_adv
                torch.save(model.state_dict(), self.default_path)
        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def get_manifold_loss(self, anchors, knns, anchors_idx):
        dis = self.graph.get_knn_distance(anchors_idx.cpu().numpy())
        dis = torch.from_numpy(dis).type_as(anchors)
        sim = torch.exp(-(dis ** 2) / self.temperature)
        anc = anchors.unsqueeze(1).repeat(1, knns.size(1), 1)
        dif = ((anc - knns) ** 2).sum(dim=2)
        loss = (sim * dif).sum() / (self.anchor_number * self.knn)
        # self.writer.add_histogram(self.similarity_tag, sim.detach().cpu().numpy())
        return loss

    def epoch_train(self, model, opt, attack, **kwargs):
        total_loss, total_acc_cln, total_acc_adv = 0., 0., 0.
        model.train()
        for idx, (x, y) in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            delta = attack(model, x, y, **kwargs, label_leaking=False)
            logits_cln, features_cln = model(x, only_logits=False)
            logits_adv, features_adv = model(x + delta, only_logits=False)
            
            self.anchor_number = idx.size(0) // (self.knn + 1)
            anchor_idx = idx[:self.anchor_number]
            anchor_cln = logits_cln[:self.anchor_number]
            anchor_adv = logits_adv[:self.anchor_number]
            knn_idx = idx[self.anchor_number:].view(self.anchor_number, self.knn)
            knn_cln = logits_cln[self.anchor_number:].view(self.anchor_number, self.knn, -1)
            knn_adv = logits_adv[self.anchor_number:].view(self.anchor_number, self.knn, -1)
            
            loss_manifold_1 = self.get_manifold_loss(anchor_cln, knn_adv, anchor_idx)
            loss_manifold_2 = self.get_manifold_loss(anchor_adv, knn_cln, anchor_idx)
            loss_manifold_3 = self.get_manifold_loss(anchor_cln, knn_cln, anchor_idx)
            loss_manifold_4 = self.get_manifold_loss(anchor_adv, knn_adv, anchor_idx)
            loss_manifold = loss_manifold_1 + loss_manifold_2 + loss_manifold_3 + loss_manifold_4

            logits_cln = logits_cln[:self.anchor_number]
            logits_adv = logits_adv[:self.anchor_number]
            y = y[:self.anchor_number]
            
            loss_cln = nn.CrossEntropyLoss()(logits_cln, y)
            loss_adv = nn.CrossEntropyLoss()(logits_adv, y)
        
            loss = (loss_cln + loss_adv) + self.lambda_item * loss_manifold
        
            opt.zero_grad()
            loss.backward()
            opt.step()
        
            total_loss += loss.item() * x.shape[0]
            total_acc_cln += (logits_cln.max(dim=1)[1] == y).sum().item()
            total_acc_adv += (logits_adv.max(dim=1)[1] == y).sum().item()
    
        total_loss /= len(self.train_loader.dataset)
        total_acc_cln /= len(self.train_loader.dataset)
        total_acc_adv /= len(self.train_loader.dataset)
        return total_loss, total_acc_cln, total_acc_adv

    def epoch_val(self, model, attack, **kwargs):
        total_loss, total_acc_cln, total_acc_adv = 0., 0., 0.
        model.eval()
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            delta = attack(model, x, y, **kwargs, label_leaking=True)
            logits_cln = model(x)
            logits_adv = model(x + delta)
            loss_cln = nn.CrossEntropyLoss()(logits_cln, y)
            loss_adv = nn.CrossEntropyLoss()(logits_adv, y)
            loss = (loss_cln + loss_adv)
            total_loss += loss.item() * x.shape[0]
            total_acc_cln += (logits_cln.max(dim=1)[1] == y).sum().item()
            total_acc_adv += (logits_adv.max(dim=1)[1] == y).sum().item()
    
        total_loss /= len(self.val_loader.dataset)
        total_acc_cln /= len(self.val_loader.dataset)
        total_acc_adv /= len(self.val_loader.dataset)
        return total_loss, total_acc_cln, total_acc_adv
