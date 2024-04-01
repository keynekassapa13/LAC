from loguru import logger
import torch
import numpy as np
from numba import jit
from torch.autograd import Function

from .mod import MOD

"""
References:
[1] Haresh, S., Kumar, S., Coskun, H., Syed, S. N., Konin, A., Zia, M. Z., & Tran, Q.-H. (2023). 
Learning by Aligning Videos in Time.
[2] https://github.com/trquhuytin/LAV-CVPR21/tree/main
"""

class LAV(MOD):
    def __init__(self, cfg):
        self.cfg = cfg
        self.criterion = self.create_criterion()
        self.shape = ()

    def create_criterion(self):
        criterion = self.lav_loss
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
        return embed
    
    def calculate_loss(self, embed, steps, seq_lens, masks=None):
        embed = self.adjust_embed(embed)
        a_embs, b_embs = torch.split(embed, self.shape[0], dim=0)
        loss = self.criterion(
            a_embs, b_embs,
            a_idx=steps[:, 0, :], b_idx=steps[:, 1, :],
            a_len=seq_lens[:, 0], b_len=seq_lens[:, 1],
            alpha=self.cfg.loss.alpha, sigma=self.cfg.loss.sigma,
            margin=self.cfg.loss.margin, num_frames=self.cfg.data_loader.num_frames,
            dtw_gamma=self.cfg.loss.dtw_gamma, dtw_normalize=self.cfg.loss.dtw_normalize
        )
        return loss
    
    def calc_distance_matrix(self, x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        dist = torch.pow(x - y, 2).sum(3)
        return dist

    def contrastive_idm(self, dist_N, idx_N, seq_len_N, sigma, margin):

        N = dist_N.size(0)
        sum_idm_N = torch.zeros(N)
        idm_N = torch.zeros(N, self.cfg.data_loader.num_frames, self.cfg.data_loader.num_frames)

        for i in range(N):
            dist = dist_N[i]
            idx = idx_N[i]
            seq_len = seq_len_N[i]

            grid_x, grid_y = torch.meshgrid(idx, idx, indexing='ij')

            prob = torch.nn.functional.relu(margin - dist)

            weights_orig = 1 + torch.pow(grid_x - grid_y, 2)

            diff = torch.abs(grid_x - grid_y) - (sigma / seq_len)
            
            _ones = torch.ones_like(diff)
            _zeros = torch.zeros_like(diff)
            weights_neg = torch.where(diff > 0, weights_orig, _zeros)

            weights_pos = torch.where(diff > 0, _zeros, _ones)

            idm = weights_neg * prob + weights_pos * dist

            sum_idm = torch.sum(idm)

            sum_idm_N[i] = sum_idm
            idm_N[i, :, :] = idm

        return sum_idm_N.cuda(), idm_N

    def lav_loss(
            self, a_emb, b_emb, 
            a_idx, b_idx, 
            a_len, b_len, 
            alpha=0.5, sigma=10, 
            margin=2, num_frames=32, 
            dtw_gamma=0.1, dtw_normalize=False):
        
        dtw_loss = SoftDTW(gamma=dtw_gamma, normalize=dtw_normalize)

        pos_loss = dtw_loss(a_emb, b_emb)

        # frame level loss
        dist_a = self.calc_distance_matrix(a_emb, a_emb)
        dist_b = self.calc_distance_matrix(b_emb, b_emb)

        idm_a, _ = self.contrastive_idm(dist_a, a_idx/a_len.view(-1, 1), a_len, sigma=sigma, margin=margin)
        idm_b, _ = self.contrastive_idm(dist_b, b_idx/b_len.view(-1, 1), b_len, sigma=sigma, margin=margin)

        total_loss = pos_loss
        total_loss = pos_loss + alpha * (idm_a + idm_b)
        total_loss = total_loss / num_frames
        total_loss = torch.mean(total_loss)

        return total_loss
    
# @jit(nopython = True)
def compute_softdtw(D, gamma):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for k in range(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                r0 = -R[k, i - 1, j - 1] / gamma
                r1 = -R[k, i - 1, j] / gamma
                r2 = -R[k, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[k, i, j] = D[k, i - 1, j - 1] + softmin
    return R

# @jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, : , -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]

class _SoftDTW(Function):
    @staticmethod
    def forward(ctx, D, gamma):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        R = torch.Tensor(compute_softdtw(D_, g_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_)).to(dev).type(dtype)
        # logger.info(f"E: {E.shape}, ")
        # tmp = grad_output.view(-1, 1, 1).expand_as(E) * E
        # logger.info(f"tmp: {tmp.shape}, ")
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None
    
class SoftDTW(torch.nn.Module):
    def __init__(self, gamma=1.0, normalize=False):
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma=gamma
        self.func_dtw = _SoftDTW.apply

    def calc_distance_matrix(self, x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        dist = torch.pow(x - y, 2).sum(3)
        return dist

    def forward(self, x, y):
        assert len(x.shape) == len(y.shape)
        squeeze = False
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            squeeze = True
        if self.normalize:
            D_xy = self.calc_distance_matrix(x, y)
            
            out_xy = self.func_dtw(D_xy, self.gamma)
            D_xx = self.calc_distance_matrix(x, x)
            out_xx = self.func_dtw(D_xx, self.gamma)
            D_yy = self.calc_distance_matrix(y, y)
            out_yy = self.func_dtw(D_yy, self.gamma)
            result = out_xy - 1/2 * (out_xx + out_yy) # distance
        else:
            D_xy = self.calc_distance_matrix(x, y)
            
            out_xy = self.func_dtw(D_xy, self.gamma)
            result = out_xy # discrepancy
        return result.squeeze(0) if squeeze else result