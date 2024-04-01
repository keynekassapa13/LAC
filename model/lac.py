import json
import torch
import torch.nn.functional as F

from .mod import MOD
from loguru import logger

def safe_div(a, b):
    out = a / b
    out[torch.isnan(out)] = 0
    return out

class LAC(MOD):
    def __init__(self, cfg):
        self.cfg = cfg
        self.criterion = self.create_criterion()

        N, M = cfg.data_loader.num_frames, cfg.data_loader.num_frames
        self.go = torch.ones((N, M), requires_grad=True)
        self.ge = torch.ones((N, M), requires_grad=True)
        # self.go = torch.tensor(1., requires_grad=True)
        # self.ge = torch.tensor(1., requires_grad=True)

        self.shape = ()

    def create_criterion(self):
        criterion = self.compute_lac_loss
        return criterion
    
    def create_optimizer(self, model_param):
        optimizer = torch.optim.Adam([
            {'params': model_param, 'lr': self.cfg.optimizer.lr},
            {'params': self.go, 'lr': 0.1},
            {'params': self.ge, 'lr': 0.1}
        ],
            betas=(0.9, 0.999),
            weight_decay=self.cfg.optimizer.weight_decay
        )
        return optimizer

    def adjust_frames(self, frames):
        N, M, T, C, W, H = frames.shape
        self.shape = (N, M, T, C, W, H)

        frames = frames.reshape(N*M, T, C, W, H)
        return frames
    
    def adjust_embed(self, embed):
        embed = embed.reshape(self.shape[0], self.shape[1], self.shape[2], -1)
        return embed
    
    def adjust_masks(self, masks):
        masks = masks.reshape(-1, 1, self.shape[2])
        return masks
    
    def calculate_loss(self, embed, steps, seq_lens, masks=None):
        embed = self.adjust_embed(embed)
        loss = self.compute_lac_loss(
                embed, seq_lens, steps, 
                self.go, self.ge, masks=masks.view(-1, 1, self.shape[2]))
        return loss
    
    def compute_lac_loss(self, embs, seq_lens, steps, go, ge, masks=None):
        var = self.cfg.loss.var
        temperature = self.cfg.loss.temperature
        sw_bool = self.cfg.loss.sw_bool
        alpha = self.cfg.loss.alpha

        B, V, T, C = embs.shape
        embs = embs.view(B*V, T, C)
        e1 = embs[0].unsqueeze(0)
        e2 = embs[1].unsqueeze(0)

        if sw_bool:
            # Local Alignment Constraint
            sw_loss = SoftSW(go, ge, temperature=temperature)
            sw_12, logits_12 = sw_loss(e1, e2)

        # Contrastive Loss
        logits = torch.matmul(e1.squeeze(0), e2.squeeze(0).T) / temperature

        view1 = steps[0, 0, :]
        view2 = steps[0, 1, :]
        view1 = view1.view(-1, 1)
        view2 = view2.view(1, -1)

        dist = torch.abs(view1 - view2)

        # pos
        pos_weight = torch.exp(-torch.square(dist)/(2*var)).type_as(logits)
        label = safe_div(pos_weight, pos_weight.sum(dim=1, keepdim=True))

        # neg
        exp_logits = torch.exp(logits)
        sum_negative = torch.sum(exp_logits, dim=1, keepdim=True)

        loss = F.kl_div(torch.log(safe_div(exp_logits, sum_negative) + 1e-6), label, reduction="none")
        sh = loss.shape
        if sw_bool:
            loss = torch.sum(loss) + alpha * (sw_12)
        else:
            loss = torch.sum(loss)
        loss = loss / (sh[0]*sh[1])

        return loss


class _SoftSW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S_xy, go, ge, temperature=1.0):

        S_xy_ = S_xy.detach().cpu().numpy()

        N, M = S_xy.shape
        D = torch.zeros((N + 1, M + 1), device=S_xy.device)
        D_p = torch.zeros((N + 2, M + 2, 4), device=S_xy.device)

        Ix = torch.zeros((N + 1, M + 1), device=S_xy.device)
        Ix_p = torch.zeros((N + 2, M + 2, 2), device=S_xy.device)

        Iy = torch.zeros((N + 1, M + 1), device=S_xy.device)
        Iy_p = torch.zeros((N + 2, M + 2, 3), device=S_xy.device)

        float_max = float('inf')

        if temperature > 0:
            for mat in (D, Ix, Iy):
                mat[0, :] = mat[:, 0] = -float_max

        def smooth_max(x, t=1.0):
            return t * torch.logsumexp(x / t, 0)

        for i in range(1, N+1):
            for j in range(1, M+1):
                D[i, j] = S_xy[i-1, j-1] + smooth_max(torch.tensor([0, D[i-1, j-1], Ix[i-1, j-1], Iy[i-1, j-1]]), temperature)
                D_p[i, j] = torch.softmax(torch.tensor([0, D[i-1, j-1], Ix[i-1, j-1], Iy[i-1, j-1]]), 0)
                
                Ix[i, j] = smooth_max(torch.tensor([D[i, j-1] - go[i-1, j-1], Ix[i, j-1] - ge[i-1, j-1]]), temperature)
                Ix_p[i, j] = torch.softmax(torch.tensor([D[i, j-1] - go[i-1, j-1], Ix[i, j-1] - ge[i-1, j-1]]), 0)

                Iy[i, j] = smooth_max(torch.tensor([D[i-1, j] - go[i-1, j-1], Ix[i-1, j] - go[i-1, j-1], Iy[i-1, j] - ge[i-1, j-1]]), temperature)
                Iy_p[i, j] = torch.softmax(torch.tensor([D[i-1, j] - go[i-1, j-1], Ix[i-1, j] - go[i-1, j-1], Iy[i-1, j] - ge[i-1, j-1]]), 0)
                
        D_ravel = torch.tensor(D.ravel())
        value = smooth_max(D_ravel, temperature)
        probas = torch.softmax(D_ravel, 0)
        probas = probas.reshape(D.shape)

        ctx.save_for_backward(
            S_xy, 
            D_p, 
            Ix_p, 
            Iy_p, 
            probas
        )
        ctx.temperature = temperature

        return value, D[1:, 1:]
    
    @staticmethod
    def backward(ctx, value_fw, probas_output):
        S_xy, D_p, Ix_p, Iy_p, probas = ctx.saved_tensors
        temperature = ctx.temperature

        N, M = S_xy.shape
        grad_D = torch.zeros((N + 2, M + 2), device=S_xy.device)
        grad_Ix = torch.zeros((N + 2, M + 2), device=S_xy.device)
        grad_Iy = torch.zeros((N + 2, M + 2), device=S_xy.device)

        for j in reversed(range(1, M + 1)):
            for i in reversed(range(1, N + 1)):
                grad_Iy[i, j] = (grad_D[i+1, j+1] * D_p[i+1, j+1, 3] + grad_Iy[i+1, j] * Iy_p[i+1, j, 2])
                grad_Ix[i, j] = (grad_D[i+1, j+1] * D_p[i+1, j+1, 2] + grad_Ix[i, j+1] * Ix_p[i, j+1, 1] + grad_Iy[i+1, j] * Iy_p[i+1, j, 1])
                grad_D[i, j] = (grad_D[i+1, j+1] * D_p[i+1, j+1, 1] + grad_Ix[i, j+1] * Ix_p[i, j+1, 0] + \
                                grad_Iy[i+1, j] * Iy_p[i+1, j, 0] + probas[i, j])
        
        grad_D_ = torch.zeros_like(S_xy, device=S_xy.device)
        grad_go = torch.zeros_like(S_xy, device=S_xy.device)
        grad_ge = torch.zeros_like(S_xy, device=S_xy.device)

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                grad_D_[i-1, j-1] = grad_D[i, j]
                grad_go[i-1, j-1] = (grad_Ix[i, j+1] * (-Ix_p[i, j+1, 0]) + grad_Iy[i+1, j] * (-Iy_p[i+1, j, 0] - Iy_p[i+1, j, 1]))
                grad_ge[i-1, j-1] = (grad_Ix[i, j+1] * (-Ix_p[i, j+1, 1]) + grad_Iy[i+1, j] * (-Iy_p[i+1, j, 2]))
                
        return value_fw.cuda().view(-1, 1).expand_as(grad_D_) * grad_D_, grad_go.cpu(), grad_ge.cpu(), None
                
class SoftSW(torch.nn.Module):
    def __init__(self, go, ge, temperature=1.0):
        super(SoftSW, self).__init__()
        self.go = go
        self.ge = ge
        self.temperature = temperature
        self.func_dtw = _SoftSW.apply

    def calc_cosine_distance_matrix(self, x, y):
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        distance = 1 - torch.nn.functional.cosine_similarity(x, y, dim=3)
        return torch.exp(distance)

    def calc_distance_matrix(self, x, y):
        # return torch.matmul(x.squeeze(0), y.squeeze(0).T).unsqueeze(0)
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        dist = torch.pow(x - y, 2).sum(3)
        return dist
    
    def forward(self, x, y):
        # Similarity matrix
        S_xy = self.calc_distance_matrix(x, y)

        B, N, M = S_xy.shape
        loss = 0
        probas = []
        for b in range(B):
            M, D = self.func_dtw(S_xy[b], self.go, self.ge, self.temperature)
            loss = loss + M
            probas.append(D)
        return loss / B, torch.stack(probas, 0)