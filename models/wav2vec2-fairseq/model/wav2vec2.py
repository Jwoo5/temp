import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.quantization import GumbelVectorQuantizer
from utils.mask import compute_mask_indices
from utils.buffered_arange import buffered_arange
from utils.grad_multiply import GradMultiply
from utils.fp32_layer_norm import LayerNorm
from modules.transformer import TransformerEncoder
from modules.layers import ConvFeatureExtraction

class Wav2Vec2(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

        feature_enc_layers = eval(cfgs.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtraction(
            conv_layers = feature_enc_layers,
            in_d = 1 if cfgs.dataset == 'mitbih' else 12,
            dropout = 0.0,
            mode = cfgs.extractor_mode,
            conv_bias = cfgs.conv_bias
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfgs.embed_dim)
            if self.embed != cfgs.embed_dim and not cfgs.quantize_input
            else None
        )

        self.mask_prob = cfgs.mask_prob
        self.mask_selection = cfgs.mask_selection
        self.mask_other = cfgs.mask_other
        self.mask_length = cfgs.mask_length
        self.no_mask_overlap = cfgs.no_mask_overlap
        self.mask_min_space = cfgs.mask_min_space

        self.mask_channel_prob = cfgs.mask_channel_prob
        self.mask_channel_selection = cfgs.mask_channel_selection
        self.mask_channel_other = cfgs.mask_channel_other
        self.mask_channel_length = cfgs.mask_channel_length
        self.no_mask_channel_overlap = cfgs.no_mask_channel_overlap
        self.mask_channel_min_space = cfgs.mask_channel_min_space

        #XXX
        self.dropout_input = nn.Dropout(cfgs.dropout_input)
        self.dropout_features = nn.Dropout(cfgs.dropout_features)

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfgs.num_negatives
        self.cross_sample_negatives = cfgs.cross_sample_negatives
        self.codebook_negatives = cfgs.codebook_negatives
        self.negatives_from_everywhere = cfgs.negatives_from_everywhere

        self.logit_temp = cfgs.logit_temp

        self.feature_grad_mult = cfgs.feature_grad_mult

        final_dim = cfgs.final_dim if cfgs.final_dim > 0 else cfgs.embed_dim

        if cfgs.quantize_targets:
            vq_dim = cfgs.latent_dim if cfgs.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfgs.latent_vars,
                temp=cfgs.latent_temp,
                groups=cfgs.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)
        
        if cfgs.quantize_input:
            if cfgs.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfgs.latent_dim if cfgs.latent_dim > 0 else cfgs.embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfgs.latent_vars,
                    temp=cfgs.latent_temp,
                    groups=cfgs.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project_inp = nn.Linear(vq_dim, self.embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfgs.embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfgs)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfgs.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )
        
        self.final_proj = nn.Linear(cfgs.embed_dim, final_dim)

    @classmethod
    def build_model(cls, cfgs, task=None):

        return cls(cfgs)
    
    def apply_mask(self, x, padding_mask):
        batch_size, timestep, channel = x.shape
        
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (batch_size, timestep),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks = 2,
                no_overlap = self.no_mask_overlap,
                min_space = self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None
        
        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (batch_size, channel),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap = self.no_mask_channel_overlap,
                min_space = self.mask_channel_min_space
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0
        
        return x, mask_indices

    def sample_negatives(self, y, num):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)
        
        batch_size, time_size, feature_size = y.shape
        y = y.view(-1, feature_size) # B x T x C -> (B x T) x C

        cross_high = time_size * batch_size
        high = time_size
        with torch.no_grad():
            assert high > 1, f"{batch_size, time_size, feature_size}"

            if self.n_negatives > 0:
                time_sizes = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )
                neg_idxs = torch.randint(
                    low = 0, high = high - 1, size = (batch_size, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= time_sizes] += 1
 
            if self.cross_sample_negatives > 0:
                time_sizes = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low = 0,
                    high = cross_high - 1,
                    size = (batch_size, self.cross_sample_negatives * num)
                )
                cross_neg_idxs[cross_neg_idxs >= time_sizes] += 1
        
        if self.n_negatives > 0:
            for i in range(1, batch_size):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim =1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            batch_size, num, self.n_negatives + self.cross_sample_negatives, feature_size
        ).permute(
            2, 0, 1, 3
        ) # to N x B x T x C

        return negs, neg_idxs
    
    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim= 0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim = -1).type_as(x)

        logits /= self.logit_temp

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits
    
    def forward(self, source, padding_mask = None, mask = True, features_only = False):
        
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        
        features_pen = features.float().pow(2).mean()

        features = features.transpose(1,2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        # if padding_mask is not None:
        #     input_lengths = (1 - padding_mask.long()).sum(-1)
        #     # apply conv formula to get real output_lengths

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        
        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features            

        else:
            x = features
            y = unmasked_features
            mask_indices = None
        
        x = self.encoder(x, padding_mask = padding_mask)

        if features_only:
            return {"x": x, "padding_mask" : padding_mask}
        
        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(unmasked_features, produce_targets = False)
                negs, _ = sample_negatives(neg_cands, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                ) # order does not matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim = 0)
            else:
                y = self.project_q(y)

                if self.negatives_from_everywhere:
                    negs, _ = self.sample_negatives(unmasked_features, y.size(1))
                    negs = self.project_q(negs)
                else:
                    negs, _ = self.sample_negatives(y, y.size(1))

            x = x[mask_indices].view(x.size(0), -1, x.size(-1))

            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)

            x = self.final_proj(x)
            x = self.compute_preds(x, y, negs)

        result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result
    
    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1,2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False):
        res = self.forward(source, padding_mask, mask = mask, features_only = True)
        return res["x"], res["padding_mask"]
    
    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0,2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits
    
    def get_targets(self, sample, net_output, expand_steps = True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype = torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None   
    
    def save(self, epoch, path):
        torch.save({
            "epoch" : epoch,
            "state_dict" : self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"],)
        return save["epoch"]