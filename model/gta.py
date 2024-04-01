import torch
import numpy as np
import torch.nn.functional as F
from numba import jit
from torch.autograd import Function

from .mod import MOD
from loguru import logger

"""
References:
[1] Hadji, I., Derpanis, K. G., & Jepson, A. D. (2021). 
Representation Learning via Global Temporal Alignment and Cycle-Consistency.
[2] https://github.com/hadjisma/VideoAlignment/tree/master for GTA loss
[3] https://github.com/trquhuytin/LAV-CVPR21 for base model and conv embedder
"""

class GTA(MOD):
    def __init__(self, cfg):
        self.cfg = cfg
        self.criterion = self.create_criterion()
        self.shape = ()

    def create_criterion(self):
        criterion = self.gta_loss
        return criterion
    
    def create_optimizer(self, model_param):
        optimizer = torch.optim.Adam(
            model_param,
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay
        )
        return optimizer

    def adjust_frames(self, frames):
        N, M, T, C, W, H = frames.shape
        self.shape = (N, M, T, C, W, H)

        frames = frames.reshape(N*M, T, C, W, H)
        return frames
    
    def adjust_steps(self, steps):
        return steps
    
    def adjust_seq_lens(self, seq_lens):
        return seq_lens
    
    def adjust_embed(self, embed):
        embed = F.normalize(embed, p=2, dim=-1)
        return embed
    
    def calculate_loss(self, embed, steps, seq_lens, masks=None):
        embed = self.adjust_embed(embed)
        loss = self.criterion(
            embed, self.shape[0] * self.shape[1],
            alignment_type=self.cfg.loss.alignment_type,
            loss_type=self.cfg.loss.alignment_type,
            similarity_type=self.cfg.loss.similarity_type,
            label_smoothing=self.cfg.loss.label_smoothing,
            softning=self.cfg.loss.softning,
            gamma_s=self.cfg.loss.gamma_s,
            gamma_f=self.cfg.loss.gamma_f
        ).to(self.cfg.device)
        return loss
    
    def add_regularization(self, loss, model):
        reg_loss = torch.mean(torch.stack(
            [torch.mean(p**2) for p in model.parameters()]))
        alpha = 1.0
        return loss + alpha * reg_loss.to(self.cfg.device)

    def gta_loss(
        self, embs, batch_size,
        alignment_type="D2TW_consistency",
        loss_type="D2TW_consistency",
        similarity_type="cosine",
        label_smoothing=0.1,
        softning="dtw_prob",
        gamma_s=0.1,
        gamma_f=0.1
    ):
        if alignment_type == "D2TW_consistency":
            loss = self.dtw_alignment_consistency_loss(
                embs, batch_size,
                loss_type, similarity_type,
                softning, gamma_s, gamma_f,
                label_smoothing
            )
        elif alignment_type == "D2TW":
            loss = self.dtw_alignment_loss(
                embs, batch_size,
                loss_type, similarity_type,
                softning, gamma_s, gamma_f
            )
        else:
            raise NotImplementedError
        
        return loss
    
    def classification_loss(self, logits, labels, label_smoothing):
        labels = labels.detach()
        return torch.mean(F.cross_entropy(logits, labels, reduction='mean', label_smoothing=label_smoothing))
    
    def dtw_alignment_consistency_loss(
            self, embs, batch_size,
            loss_type, similarity_type,
            softning, gamma_s, gamma_f,
            label_smoothing
    ):
        logits_list = []
        logits_ij_list = []
        logits_ji_list = []
        labels_list = []

        i = 0
        skip = 1

        for j in range(i+1, batch_size):
            logits_ij, _ = smoothDTW(
                embs[i,::skip,:], embs[j],
                similarity_type, softning, gamma_s, gamma_f)
            logits_ij_list.append(logits_ij[-1, -1])
            logits_ij = F.softmax(-logits_ij[1:, 1:], dim=0)

            logits_ji, _ = smoothDTW(
                embs[j], embs[i,::skip,:],
                similarity_type, softning, gamma_s, gamma_f)
            logits_ji_list.append(logits_ji[-1, -1])
            logits_ji = F.softmax(-logits_ji[1:, 1:], dim=0)

            logits = torch.matmul(logits_ij, logits_ji)
            logits_list.append(logits)
            labels = torch.eye(logits.shape[0])
            labels_list.append(labels)

        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        logits_ij_list = torch.stack(logits_ij_list)
        logits_ji_list = torch.stack(logits_ji_list)

        loss_sdtw_ij = torch.mean(logits_ij_list)
        loss_sdtw_ji = torch.mean(logits_ji_list)

        loss_con = self.classification_loss(logits, labels, label_smoothing)
        loss = loss_con + 0.1 * loss_sdtw_ij + 0.1 * loss_sdtw_ji
        return loss
    
    def dtw_alignment_loss(
            self, embs, batch_size,
            loss_type, similarity_type,
            softning, gamma_s, gamma_f
    ):
        logits_list = []
        i = 0
        for j in range(i+1, batch_size):
            logits, _ = smoothDTW(
                embs[i], embs[j],
                similarity_type, softning, gamma_s, gamma_f)
            logits_list.append(logits[-1, -1])

        logits = torch.stack(logits_list, dim=0)
        # Non discriminative DTW loss
        loss = torch.mean(logits)
        return loss
        

def minGamma(neighbors, gamma=1):
    if gamma == 0: 
        minG = torch.min(neighbors)
    else:
        zi = (-neighbors) / gamma
        max_zi = torch.max(zi)
        log_sum_G = max_zi + torch.log(torch.sum(torch.exp(zi - max_zi)))
        minG = - gamma * log_sum_G
    return minG
    
def smoothDTW(embs1, embs2, distance_type, softning, gamma_s, gamma_f):
    if distance_type == 'cosine':
        dist = torch.matmul(embs1, embs2.T)
    else:
        raise NotImplementedError
    
    # normalize distance column wise
    dist = - torch.log(F.softmax(dist/gamma_f, dim=0))

    nrows, ncols = dist.shape

    sdtw = torch.zeros((nrows+1, ncols+1), dtype=torch.float32)
    for i in range(0, nrows + 1):
        for j in range(0, ncols + 1):
            if (i==0) and (j==0):
                new_val = torch.tensor(0.0, dtype=torch.float32)
                sdtw[i, j] = new_val
            elif (i==0) and (j!=0):
                new_val = torch.finfo(torch.float32).max
                sdtw[i, j] = new_val
            elif (i!=0) and (j==0):
                new_val = torch.finfo(torch.float32).max
                sdtw[i, j] = new_val
            else:
                neighbors = torch.stack([
                    sdtw[i, j-1],
                    sdtw[i-1, j-1],
                    sdtw[i-1, j]
                ])

                if softning == 'dtw_minGamma':
                    new_val = dist[i-1, j-1] + minGamma(neighbors, gamma_s)
                    sdtw[i, j] = new_val

                elif softning == 'dtw_prob':
                    probs = F.softmax(-neighbors/gamma_s, dim=-1)
                    new_val = dist[i-1, j-1] + (probs[0] * sdtw[i, j-1]) + (probs[1] * sdtw[i-1, j-1]) + (probs[2] * sdtw[i-1, j])
                    
                    # sdtw[i, j] = new_val

                    sdtw_clone = sdtw.clone()
                    sdtw_clone[i, j] = new_val
                    sdtw = sdtw_clone

                elif softning == "non-diff":
                    new_val = dist[i-1, j-1] + torch.min(torch.tensor([
                        sdtw[i, j-1], sdtw[i-1, j-1], sdtw[i-1, j]
                    ]))
                    sdtw[i, j] = new_val
                else:
                    raise NotImplementedError
    return sdtw, dist