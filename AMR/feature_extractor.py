import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils
import os
import time
import datetime
from tensorboardX import SummaryWriter

from loader import get_loader, get_extractor_model


class FeatureExtractor(object):
    """Low-dimensional embedding for image
    
    Args:
        method (string): The choice of extractor.
            Possible values: 'AE'(auto-encoder), 'NT'(natural-trained classifier)
            or 'AT'(adversarial-trained classifier).
    """
    def __init__(self, config):
        self.dataset = config.dataset
        self.method = config.extractor      # The choice of extractor
        assert self.method == 'NT' or self.method == 'AT' or self.method == 'AE'
        self.epochs = config.extractor_epochs
        self.extractor_save_dir = config.extractor_save_dir
        self.data_dir = config.data_dir
        
        self.data_loader = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(log_dir=os.path.join(config.log_dir, self.dataset.lower()))
        
        self.default_name = '{}-{}.pt'.format(self.dataset.lower(), self.method.lower())
        self.default_path = os.path.join(self.extractor_save_dir, self.default_name)
        self.feature_name = '{}-{}-features.pt'.format(self.dataset.lower(), self.method.lower())
        self.feature_path = os.path.join(self.extractor_save_dir, self.feature_name)

    def _train_nt(self):
        """Natural training"""
        opt = optim.Adam(self.model.parameters())
        start_time = time.time()
        for epoch_idx in range(self.epochs):
            total_loss, total_acc = 0., 0.
            for _, (X, y) in self.data_loader:
                x, y = X.to(self.device), y.to(self.device)
                yp = self.model(x)
                loss = nn.CrossEntropyLoss()(yp, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * x.size(0)
                total_acc += (yp.max(dim=1)[1] == y).sum().item()
            avg_loss, avg_acc = total_loss / len(self.data_loader.dataset), total_acc / len(self.data_loader.dataset)
            
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = 'Elapsed [{}], Epoch [{}/{}]'.format(et, epoch_idx + 1, self.epochs)
            log += ', Loss: {:.4f}'.format(avg_loss)
            log += ', Accuracy: {:.4f}'.format(avg_acc)
            print(log)
            self.writer.add_scalar('EXTRACTOR/NT/LOSS', avg_loss, epoch_idx)
            self.writer.add_scalar('EXTRACTOR/NT/ACC', avg_acc, epoch_idx)
    
    def _train_at(self):
        """Adversarial training"""
        # Lazy import
        from attack_utils import attack_configs, attack_function
        attack = attack_function['FGSM']
        kwargs = attack_configs['FGSM'][self.dataset]
        opt = optim.Adam(self.model.parameters())
        start_time = time.time()
        for epoch_idx in range(self.epochs):
            total_loss, total_acc_cln, total_acc_adv = 0., 0., 0.
            for _, (X, y) in self.data_loader:
                x, y = X.to(self.device), y.to(self.device)
                delta = attack(self.model, x, y, **kwargs)
                logits_cln = self.model(x)
                logits_adv = self.model(x + delta)
                loss_cln = nn.CrossEntropyLoss()(logits_cln, y)
                loss_adv = nn.CrossEntropyLoss()(logits_adv, y)
                loss = loss_cln + loss_adv
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * x.size(0)
                total_acc_cln += (logits_cln.max(dim=1)[1] == y).sum().item()
                total_acc_adv += (logits_adv.max(dim=1)[1] == y).sum().item()
            avg_loss, avg_acc_cln, avg_acc_adv = total_loss / len(self.data_loader.dataset), \
                                                 total_acc_cln / len(self.data_loader.dataset), \
                                                 total_acc_cln / len(self.data_loader.dataset)
    
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = 'Elapsed [{}], Epoch [{}/{}]'.format(et, epoch_idx + 1, self.epochs)
            log += ', Loss: {:.4f}'.format(avg_loss)
            log += ', Acc_cln: {:.4f}'.format(avg_acc_cln)
            log += ', Acc_adv: {:.4f}'.format(avg_acc_adv)
            print(log)
            self.writer.add_scalar('EXTRACTOR/AT/LOSS', avg_loss, epoch_idx)
            self.writer.add_scalars('EXTRACTOR/AT/ACC', {
                'cln': avg_acc_cln,
                'adv': avg_acc_adv,
            }, epoch_idx)
    
    def _train_ae(self):
        """Auto-encoder training"""
        opt = optim.Adam(self.model.parameters())
        start_time = time.time()
        for epoch_idx in range(self.epochs):
            total_loss = 0.
            x, x_rec = None, None
            for _, (X, y) in self.data_loader:
                x = X.to(self.device)
                x_rec = self.model(x)
                loss = nn.MSELoss()(x, x_rec)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * x.size(0)
            avg_loss = total_loss / len(self.data_loader.dataset)

            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = 'Elapsed [{}], Epoch [{}/{}]'.format(et, epoch_idx + 1, self.epochs)
            log += ', Loss: {:.4f}'.format(avg_loss)
            print(log)
            
            # Visualize the recovered images
            imgs = torch.cat((x[:8].cpu(), x_rec[:8].cpu()), dim=0)
            imgs = vutils.make_grid(imgs, normalize=True, scale_each=True)
            self.writer.add_image('Image', imgs, epoch_idx)
            self.writer.add_scalar('EXTRACTOR/AE/LOSS', avg_loss, epoch_idx)
        
    def train_extractor(self, batch_size=128):
        self.data_loader = get_loader(self.dataset, self.data_dir, batch_size=batch_size, mode='train', shuffle=True)
        self.model = get_extractor_model(dataset=self.dataset, method=self.method)
        if self.method == 'NT':
            self._train_nt()
        elif self.method == 'AT':
            self._train_at()
        elif self.method == 'AE':
            self._train_ae()
        torch.save(self.model.state_dict(), self.default_path)
        print('Saved extractor checkpoints into {}...'.format(self.default_path))
    
    def run(self):
        if os.path.exists(self.default_path):
            print('Using trained extractor checkpoints: {}'.format(self.default_path))
        else:
            print('Training feature extractor...')
            self.train_extractor()
        
        if os.path.exists(self.feature_path):
            print('Using computed features: {}'.format(self.feature_path))
        else:
            self.data_loader = get_loader(self.dataset, self.data_dir, batch_size=256, mode='train', shuffle=False)
            self.model = get_extractor_model(self.dataset, self.method, extractor_save_dir=self.extractor_save_dir)
            features = []
            for idx, (X, y) in self.data_loader:
                x = X.to(self.device)
                z = self.model(x, only_logits=False)[1]
                features.append(z.detach())
            tensor = torch.cat(tuple(features), dim=0)
            torch.save(tensor, self.feature_path)
            print('Saved features into {}...'.format(self.feature_path))
            
        
def test():
    import argparse
    parser = argparse.ArgumentParser()
    config = parser.parse_args()
    config.extractor = 'AE'
    config.dataset = 'MNIST'
    config.extractor_epochs = 10
    config.extractor_save_dir = './results/extractor'
    config.data_dir = '../data'
    config.log_dir = './results/logs'
    if not os.path.exists(config.extractor_save_dir):
        os.makedirs(config.extractor_save_dir)
    extractor = FeatureExtractor(config)
    extractor.run()


if __name__ == '__main__':
    test()

