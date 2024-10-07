import torch
import itertools
from copy import deepcopy
from argparse import ArgumentParser

from datasets.exemplars_dataset import ExemplarsDataset
from .incremental_learning import Inc_Learning_Appr

class Appr(Inc_Learning_Appr):
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr=1e-1, wu_fix_bn=False,
                 wu_scheduler='constant', wu_patience=None, wu_wd=0., fix_bn=False, eval_on_train=False, 
                 select_best_model_by_val_loss=True, logger=None, exemplars_dataset=None, scheduler_milestones=None,
                 lamb_lwf=1, T=2, lamb_ewc=5000, alpha=0.5, fi_sampling_type='max_pred', fi_num_samples=-1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad,
                                   momentum, wd, multi_softmax, wu_nepochs, wu_lr, wu_fix_bn,
                                   wu_scheduler, wu_patience, wu_wd, fix_bn, eval_on_train,
                                   select_best_model_by_val_loss, logger, exemplars_dataset, scheduler_milestones)
        self.model_old = None
        self.lamb_lwf = lamb_lwf
        self.T = T

        self.lamb_ewc = lamb_ewc
        self.alpha = alpha
        self.sampling_type = fi_sampling_type
        self.num_samples = fi_num_samples

        feat_ext = self.model.model
        self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros(p.shape).to(device) for n, p in feat_ext.named_parameters() if p.requires_grad}

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset
    
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # lambda lwf is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda.
        parser.add_argument('--lamb-lwf', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off in LwF (default=%(default)s)')
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        
        # lambda ewc sets how important the old task is compared to the new one"
        parser.add_argument('--lamb-ewc', default=5000, type=float, required=False,
                            help='Forgetting-intransigence trade-off in EWC (default=%(default)s)')
        parser.add_argument('--alpha', default=0.5, type=float, required=False,
                            help='EWC alpha (default=%(default)s)')
        parser.add_argument('--fi-sampling-type', default='max_pred', type=str, required=False,
                            choices=['true', 'max_pred', 'multinomial'],
                            help='Sampling type for Fisher information (default=%(default)s)')
        parser.add_argument('--fi-num-samples', default=-1, type=int, required=False,
                            help='Number of samples for Fisher information (-1: all available) (default=%(default)s)')

        return parser.parse_known_args(args)
    
    def _get_optimizer(self):
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
    
    def compute_fisher_matrix_diag(self, trn_loader):
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.model.model.named_parameters() if p.requires_grad}
        n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        self.model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            outputs = self.model.forward(images.to(self.device))

            if self.sampling_type == 'true':
                preds = targets.to(self.device)
            elif self.sampling_type == 'max_pred':
                preds = torch.cat(outputs, dim=1).argmax(1).flatten()
            elif self.sampling_type == 'multinomial':
                probs = torch.nn.functional.softmax(torch.cat(outputs, dim=1), dim=1)
                preds = torch.multinomial(probs, len(targets)).flatten()

        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), preds)
        self.optimizer.zero_grad()
        loss.backward()
        for n, p in self.model.model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2) * len(targets)
        n_samples = n_samples_batches * trn_loader.batch_size
        fisher = {n: (p / n_samples) for n, p in fisher.items()}        
        return fisher
    
    def train_loop(self, t, trn_loader, val_loader):
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.Dataloader(trn_loader.dataset + self.exemplars_dataset,
                                                     batchsize=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
        super().train_loop(t, trn_loader, val_loader)

        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
    
    def train_epoch(self, t, trn_loader):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            target_old = None
            if t > 0:
                target_old = self.model_old.forward(images.to(self.device))
            outputs = self.model.forward(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device), target_old)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
    
    def post_train_process(self, t, trn_loader):
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

        self.older_params = {n: p.clone().detach() for n, p in self.model.model.named_parameters() if p.requires_grad}
        curr_fisher = self.compute_fisher_matrix_diag(trn_loader)
        for n in self.fisher.keys():
            if self.alpha == -1:
                alpha = (sum(self.model.task_cls[:t]) / sum(self.model.task_cls)).to(self.device)
                self.fisher[n] = alpha * self.fisher[n] + (1 - alpha) * curr_fisher[n]
            else:
                self.fisher[n] = self.alpha * self.fisher[n] + (1 - self.alpha) * curr_fisher[n]

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            # if self.model_old is not None:
            #     self.model_old.eval()

            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images)
                # Forward current model
                outputs = self.model(images)
                loss = self.criterion(t, outputs, targets, targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num
    
    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if(size_average):
            ce = ce.mean()
        return ce
    
    def criterion(self, t, outputs, targets, outputs_old=None):
        if t > 0 and outputs_old is not None:
            # Elastic weight consolidation quadratic penalty
            loss_reg = 0
            for n, p in self.model.model.named_parameters():
                if n in self.fisher.keys():
                    loss_reg += torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2)) / 2
            # Knowledge distillation loss for all previous tasks
            kd_outputs, kd_outputs_old = torch.cat(outputs[:t], dim=1), torch.cat(outputs_old[:t], dim=1)
            loss_kd = self.cross_entropy(kd_outputs, kd_outputs_old, exp=1.0 / self.T)
        else:
            loss_reg = 0
            loss_kd = 0

        if len(self.exemplars_dataset) > 0:
            loss_ce = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        else:
            loss_ce = torch.nn.functional.cross_entropy(outputs[t], (targets - self.model.task_offset[t]).to(torch.long))

        return loss_ce + self.lamb_lwf * loss_kd + self.lamb_ewc * loss_reg
    
