import argparse

import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

import pdb
import wandb

from torch.utils.data.sampler import Sampler
from model.resnet import Resnet
from cfgs import config as cfg
from data.mitbih import MITBIHDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default = './trained_models/save.pth', type=str, required=False,
                        help="save file")
    parser.add_argument('--config', default = './cfgs/config.json', type=str, required=False,
                        help="config file")
    parser.add_argument('--epoch', default = 200, type = int, required = False,
                        help = "epoch")
    parser.add_argument('--batch', default = 256, type = int, required= False,
                        help = "batch size")
    parser.add_argument('--gpu', default = 0, type = int, required = False,
                        help = "GPU id to use")
    parser.add_argument('--nw', dest = 'num_workers', default = 0, type = int,
                        help = "number of workers to load data")

    parser.add_argument('--o', dest='optimizer', default = "adam", type = str,
                        help = 'training optimizer')
    parser.add_argument('--lr', type = float, default = 1e-3, required = False,
                        help = "learning rate")
    parser.add_argument('--lr_decay_step', dest = 'lr_decay_step', default = 10000, type = int,
                        help = 'step to do learning rate decay, unit is epoch')
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', default=0.75, type=float,
            help='learning rate decay ratio')

    args = parser.parse_args()
    return args

class Sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_batch * batch_size, train_size).long()
            self.leftover_flag = True
    
    def __iter__(self):
        rand_num = torch.randperm(self.num_of_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_of_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)
        
        return iter(self.rand_num_view)
    
    def __len__(self):
        return self.num_data

if __name__ == '__main__':

    args = parse_args()

    print("Called with args: ")
    print(args)

    config = cfg.Config.load(args.config)
    config.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # for reproduction
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    train_path = '../../data/mitdb_refined_csv/mitbih_train_balanced.csv'
    test_path = '../../data/mitdb_refined_csv/mitbih_test.csv'

    # wandb.init(project="mitbih-transformer")

    train_dataset = MITBIHDataset(train_path)
    test_dataset = MITBIHDataset(test_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch, sampler = None, shuffle = True, num_workers = args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch, sampler = None, shuffle = True, num_workers = args.num_workers)

    model = Resnet(d_in = 1, d_model = config.d_model, n_layer = config.n_layer, n_classes = config.n_classes)

    criterion_cls = torch.nn.CrossEntropyLoss()

    t_total = len(train_loader) * args.epoch
    
    lr = args.lr
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params' : [value], 'lr' : lr * 2, \
                            'weight_decay' : 0 }]
        else:
            params += [{'params':[value],'lr':lr, 'weight_decay': 0.0005}]
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)
    model.to(config.device)
    # wandb.watch(model)    

    inputs = torch.FloatTensor(1)
    label = torch.LongTensor(1)

    inputs = inputs.to(config.device)
    label = label.to(config.device)

    inputs = Variable(inputs)
    label = Variable(label)

    best_epoch, best_loss, best_score = 0, 0, 0
    if os.path.isfile(args.save):
        best_epoch, best_loss, best_score = model.load(args.save)
        print(f"rank: {config.device} load state dict from: {args.save}")

    offset = best_epoch
    for step in trange(args.epoch, desc = "Epoch"):
        epoch = step + offset
        model.train()

        losses = []
        with tqdm(total = len(train_loader), desc=f"Train({config.device}) {epoch}") as pbar:
            for i, data in enumerate(train_loader):
                with torch.no_grad():
                    inputs.resize_(data[0].size()).copy_(data[0])
                    label.resize_(data[1].size()).copy_(data[1])

                optimizer.zero_grad()
                outputs = model(inputs)
                logits_cls = outputs

                loss_cls = criterion_cls(logits_cls, label)
                loss = loss_cls

                loss_val = loss_cls.item()
                losses.append(loss_val)

                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
                
        avg_loss = np.mean(losses)
        # del...?
        # del data

        if step % 5 == 0:
            matchs = []
            model.eval()
            with tqdm(total=len(test_loader), desc=f"Valid({config.device})") as pbar:
                for i, data in enumerate(test_loader):
                    with torch.no_grad():
                        inputs.resize_(data[0].size()).copy_(data[0])
                        label.resize_(data[1].size()).copy_(data[1])

                    outputs = model(inputs)
                    logits_cls = outputs
                    _, output_cls = logits_cls.max(1)

                    match = torch.eq(output_cls, label).detach()
                    matchs.extend(match.cpu())
                    accuracy = np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

                    pbar.update(1)
                    pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
            score = accuracy
            
            # wandb.log({"loss" : avg_loss, "accuracy" : score})

            if best_score < score:
                best_epoch = step
                best_score = score
                best_loss = avg_loss
                model.save(epoch, avg_loss, score, args.save)
                print(f">>>> rank: {config.device} save model to {args.save}, epoch={best_epoch}, loss={best_loss:.3f}, score={best_score:.3f}")