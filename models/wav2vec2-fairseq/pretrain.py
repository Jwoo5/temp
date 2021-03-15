import argparse
from typing import List, Tuple

import os
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm, trange

from data.mitbih import MITBIHDataset
from data.physionet import PhysionetDataset
from model.wav2vec2 import Wav2Vec2

def parse_args():
    parser = argparse.ArgumentParser(description = 'Train a wav2vec2 model for ECG signals')

    parser.add_argument('--extractor_mode', default = 'default', type = str, required = False,
                        help = 'mode for feature extrctor.')

    # model
    parser.add_argument('--n_layers', default = 12, type = int, required = False,
                        help = 'number of encoder layers in the transformer')
    parser.add_argument('--embed_dim', default = 768, type = int, required = False,
                        help = 'encoder embedding dimension')
    parser.add_argument('--ffn_dim', default = 3072, type = int, required = False,
                        help = 'encoder embedding dimension for FFN')
    parser.add_argument('--n_heads', default = 12, type = int, required = False,
                        help = 'number of encoder attention heads')
    parser.add_argument('--activation', default = 'gelu', type = str, required = False,
                        help = 'activation function to use')

    # dropouts
    parser.add_argument('--dropout', default = 0.1, type = float, required = False,
                        help = 'dropout probability for the transformer')
    parser.add_argument('--attention_dropout', default = 0.1, type = float, required = False,
                        help = 'dropout probability for attention weights')
    parser.add_argument('--activation_dropout', default = 0.0, type = float, required = False,
                        help = 'dropout probability after activation in FFN')
    parser.add_argument('--layer_dropout', default = 0.05, type = float, required = False,
                        help = 'probability of dropping a transformer layer')
    parser.add_argument('--dropout_input', default = 0.1, type = float, required = False,
                        help = 'dropout to apply to the input (after feature extraction)')
    parser.add_argument('--dropout_features', default = 0.1, type = float, required = False,
                        help = 'dropout to apply to the features (after feature extraction)')

    parser.add_argument('--final_dim', default = 256, type = int, required = False,
                        help = 'project final representation and targets to this many dimensions.')
    parser.add_argument('--layer_norm_first', default = False, type = bool, required = False,
                        help = 'apply layernorm first in the transformer')
    parser.add_argument('--conv_feature_layers', default = "[(512,2,2)] + [(512,2,2)] + [(512,2,2)]", type = str, required = False,
                        help = 'string describing convolutional feature extraction layers in form of a python list that contains')
    parser.add_argument('--conv_bias', default = False, type = bool, required = False,
                        help = 'include bias in conv encoder')
    parser.add_argument('--logit_temp', default = 0.1, type = float, required = False,
                        help = 'temperature to divide logits by')
    parser.add_argument('--quantize_targets', default = True, type = bool, required = False,
                        help = 'use quantized targets')
    parser.add_argument('--quantize_input', default = False, type = bool, required = False,
                        help = 'use quantized inputs')
    parser.add_argument('--same_quantizer', default = False, type = bool, required = False,
                        help = 'use same quantizer for inputs and targets')
    parser.add_argument('--target_glu', default = False, type = bool, required = False,
                        help = 'adds projection + glu to targets')
    parser.add_argument('--feature_grad_mult', default = 0.1, type = float, required = False,
                        help = 'multiply feature extractor var grads by this')
    parser.add_argument('--latent_vars', default = 320, type = int, required = False,
                        help = 'number of latent variables V in each group of the codebook')
    parser.add_argument('--latent_groups', default = 2, type = int, required = False,
                        help = 'number of groups G of latent variables in the codebook')
    parser.add_argument('--latent_dim', default = 0, type = int, required = False,
                        help = 'if > 0, uses this dimensionality for latent variables. otherwise uses final_dim / latent_groups')

    # masking
    parser.add_argument('--mask_length', default = 10, type = int, required = False,
                        help = 'mask length')
    parser.add_argument('--mask_prob', default = 0.65, type = float, required = False,
                        help = 'probability of replacing a token with mask')
    parser.add_argument('--mask_selection', default = "static", type = str, required = False,
                        help = 'how to choose mask length')
    parser.add_argument('--mask_other', default = 0, type = float, required = False,
                        help = 'secondary mask argument (used for more complex distributions)')
    parser.add_argument('--no_mask_overlap', default = False, type = bool, required = False,
                        help = 'whether to allow masks to overlap')
    parser.add_argument('--mask_min_space', default = 1, type = int, required = False,
                        help = 'min space between spans (if no overlap is enabled)')

    # channel masking
    parser.add_argument('--mask_channel_length', default = 10, type = int, required = False,
                        help = 'length of the mask for features (channels)')
    parser.add_argument('--mask_channel_prob', default = 0.0, type = float, required = False,
                        help = 'probability of replacing a featuere with 0')
    parser.add_argument('--mask_channel_selection', default = "static", type = str, required = False,
                        help = 'how to choose mask length for channel masking')
    parser.add_argument('--mask_channel_other', default = 0, type = float, required = False,
                        help = 'secondary mask argument *used for more complex distributions)')
    parser.add_argument('--no_mask_channel_overlap', default = False, type = bool, required = False,
                        help = 'whether to allow channel masks to overlap')
    parser.add_argument('--mask_channel_min_space', default = 1, type = int, required = False,
                        help = 'min space between spans (if no overlap is enabled)')

    # negative selection
    parser.add_argument('--num_negatives', default = 100, type = int, required = False,
                        help = 'number of negative examples from the same sample')
    parser.add_argument('--negatives_from_everywhere', default = False, type = bool, required = False,
                        help = 'sample negatives from everywhere, not just masked states')
    parser.add_argument('--cross_sample_negatives', default = 0, type = int, required = False,
                        help = 'number of negative examples from the any sample')
    parser.add_argument('--codebook_negatives', default = 0, type = int, required = False,
                        help = 'number of negative examples codebook')

    # positional embeddings
    parser.add_argument('--conv_pos', default = 128, type = int, required = False,
                        help = 'number of filters for convolutional positional embeddings')
    parser.add_argument('--conv_pos_groups', default = 16, type = int, required = False,
                        help = 'number of groups for convolutional positional embedding')    
    parser.add_argument('--latent_temp', default = (2, 0.5, 0.999995), type = Tuple[float, float, float], required = False,
                        help = 'temperature for latent variable sampling')

    # criterion
    parser.add_argument('--loss_weights', default = (0.1, 10), type = Tuple[float, float], required = False,
                        help = 'weights for extra losses')
    
    # optimization
    parser.add_argument('--o', dest='optimizer', default = "adam", type = str,
                        help = 'training optimizer')
    parser.add_argument('--lr', type = float, default = 5e-4, required = False,
                        help = "learning rate")
    parser.add_argument('--adam_betas', type = Tuple[float, float], default = [0.9, 0.98], required = False,
                        help = "adam betas")
    parser.add_argument('--adam_eps', type = float, default = 1e-6, required = False,
                        help = "adam epsilon")
    parser.add_argument('--weight_decay', type = float, default = 0.01, required = False,
                        help = "weight decay for adam optimizer")

    # trainig options
    parser.add_argument('--dataset', default = 'physionet', type=str, required=False,
                        help="dataset")
    parser.add_argument('--save', default = './trained_models/pretrained.pth', type=str, required=False,
                        help="save file")
    parser.add_argument('--epoch', default = 200, type = int, required = False,
                        help = "epoch")
    parser.add_argument('--batch', default = 8, type = int, required= False,
                        help = "batch size")
    parser.add_argument('--gpu', default = 0, type = int, required = False,
                        help = "GPU id to use")
    parser.add_argument('--nw', dest = 'num_workers', default = 0, type = int,
                        help = "number of workers to load data")

    args = parser.parse_args()
    return args

def train(rank, world_size, args):
    if 1 < world_size:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=rank, world_size = world_size)
    master = (world_size == 0 or rank % world_size == 0)

    args.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'physionet':
        train_path = '../../data/physionet2021/'
    elif args.dataset == 'mitbih':
        train_path = '../../data/mitdb_cropped_wav/'
    else:
        raise NotImplementedError

    if args.dataset == 'physionet':
        train_dataset = PhysionetDataset(train_path)
    elif args.dataset == 'mitbih':
        train_dataset = MITBIHDataset(train_path)
    else:
        raise NotImplementedError

    model = Wav2Vec2(args)

    loss_weights = args.loss_weights

    if 1 < world_size:
        model.to(args.device)
        model = DistributedDataParallel(model, device_ids = [rank], find_unused_parameters = True)
    else:
        model.to(args.device)

    if 1 < world_size:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch, sampler = sampler)
    else:
        sampler = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch, sampler = sampler, shuffle = False)
    
    t_total = len(train_loader) * args.epoch


    lr = args.lr
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params' : [value], 'lr' : lr * 2, \
                            'weight_decay' : 0 }]
            else:
                params += [{'params' : [value], 'lr' : lr, 'weight_decay' : args.weight_decay}]
    optimizer = torch.optim.Adam(params, lr = lr, betas = args.adam_betas, eps = args.adam_eps, weight_decay = args.weight_decay)

    model.to(args.device)

    input = torch.FloatTensor(1)
    input = input.to(args.device)
    input = Variable(input)

    if os.path.isfile('./trained_models/' + args.dataset + '_pretrained.pth'):
        if isinstance(model, DistributedDataParallel):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            save = torch.load('./trained_models/' + args.dataset + '_pretrained.pth', map_location=map_location)
            start = save["epoch"]
            model.module.load_state_dict(save["state_dict"])
        else:
            start = model.load('./trained_models/' + args.dataset + '_pretrained.pth')
    else:
        start = 0

    total_step = 0
    for epoch in trange(start, args.epoch, desc = "Epoch"):
        model.train()
        total_loss = 0
        with tqdm(total = len(train_loader), desc = f"Train({rank}) {epoch}") as pbar:
            for step, data in enumerate(train_loader):
                val, _ = data
                with torch.no_grad():
                    input.resize_(val.size()).copy_(val)
                del val

                optimizer.zero_grad()

                with torch.autograd.profiler.record_function("forward"):
                    net_output = model(input)

                    if isinstance(model, DistributedDataParallel):
                        logits = model.module.get_logits(net_output).float()
                        target = model.module.get_targets(sample = input, net_output = net_output)
                    else:
                        logits = model.get_logits(net_output).float()
                        target = model.get_targets(sample = input, net_output = net_output)

                    contrastive_loss = F.cross_entropy(
                        logits,
                        target,
                        reduction = "sum"
                        )

                    sample_size = target.numel()
                    diversity_loss = 0
                    if loss_weights is not None:
                        if isinstance(model, DistributedDataParallel):
                            extra_losses = model.module.get_extra_losses(net_output)
                        else:
                            extra_losses = model.get_extra_losses(net_output)
                        if torch.is_tensor(extra_losses):
                            extra_losses = [extra_losses]
                        if len(loss_weights) == 1 and len(extra_losses) != 1:
                            loss_weights = [loss_weights[0]] * len(extra_losses)
                        
                        for p, coef in zip(extra_losses, loss_weights):
                            if coef != 0 and p is not None:
                                p = coef * p.float() * sample_size
                                diversity_loss += p

                    loss = contrastive_loss + diversity_loss

                    # loss = contrastive_loss

                with torch.autograd.profiler.record_function("backward"):
                    loss.backward()
                    optimizer.step()
                
                total_step += 1
                if isinstance(model, DistributedDataParallel):
                    model.module.quantizer.set_num_updates(total_step)
                else:
                    model.quantizer.set_num_updates(total_step)
                total_loss += loss                
                loss_mean = total_loss / (step+1)

                pbar.update(1)
                pbar.set_postfix_str(f"Loss_M : ({loss_mean:.3f}), Loss_C : ({contrastive_loss:.3f}), Loss_D : ({diversity_loss:.3f})")

                del contrastive_loss
                del diversity_loss
                del loss
                del data
        
        if master:
            if isinstance(model, DistributedDataParallel):
                model.module.save(epoch, './trained_models/' + args.dataset +'_pretrained.pth')
            else:
                model.save(epoch, './trained_models/' + args.dataset + '_pretrained.pth')
            print(f">>>> rank: {rank} save model, epoch = {epoch}")
    
    print(f">>>> rank: {rank}")
    if 1 < world_size:
        dist.destroy_process_group()

if __name__ == '__main__':

    args = parse_args()

    print("Called with args: ")
    print(args)

    if 1 < args.num_workers:
        mp.spawn(train,
            args = (args.num_workers, args),
            nprocs = args.num_workers,
            join = True)
    else:
        train(args.gpu, args.num_workers, args)

    # TODO
    # 1. save model only when loss decreased (+ total_step, ...)
    # 2. evaluate (validation for downstream tasks)