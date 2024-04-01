import torch
import torch.nn.functional as F

from .mod import MOD

"""
References:
[1] Chen, M., Wei, F., Li, C., & Cai, D. (2022). 
Frame-wise Action Representations for Long Videos via Sequence Contrastive Learning. 
[2] https://github.com/minghchen/CARL_code
"""

class SCL(MOD):
    def __init__(self, cfg):
        self.cfg = cfg
        self.criterion = self.create_criterion()
        self.shape = ()

    def create_criterion(self):
        criterion = self.compute_sequence_loss
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
    
    def adjust_embed(self, embed):
        embed = embed.reshape(self.shape[0], self.shape[1], self.shape[2], -1)
        return embed
    
    def adjust_masks(self, masks):
        masks = masks.reshape(-1, 1, self.shape[2])
        return masks
    
    def calculate_loss(self, embed, steps, seq_lens, masks=None):
        embed = self.adjust_embed(embed)
        loss = self.compute_sequence_loss(
            embed, 
            seq_lens.to(self.cfg.device), 
            steps.to(self.cfg.device), 
            masks.to(self.cfg.device)
        )
        return loss
    
    def compute_sequence_loss(self, embs, seq_lens, steps, masks=None):
        def safe_div(a, b):
            out = a / b
            out[torch.isnan(out)] = 0
            return out
        
        negative_type = self.cfg.loss.negative_type
        positive_type = self.cfg.loss.positive_type
        label_varience = self.cfg.loss.label_varience
        temperature = self.cfg.loss.temperature

        batch_size, num_views, num_frames, channels = embs.shape

        embs = embs.view(-1, channels) # (batch_size*num_views*num_frames, channels)
        steps = steps.view(-1)
        seq_lens = seq_lens.unsqueeze(-1).expand(batch_size, num_views, num_frames).contiguous().view(-1).float()
        input_masks = masks.view(-1, 1)*masks.view(1, -1)

        logits = torch.matmul(embs, embs.transpose(0,1)) / temperature
        distence = torch.abs(steps.view(-1,1)/seq_lens.view(-1,1)*seq_lens.view(1,-1)-steps.view(1,-1))
        distence.masked_fill_((input_masks==0), 1e6)
        weight = torch.ones_like(logits)
        nn = torch.zeros_like(steps).long()

        # negative weight
        for b in range(batch_size):
            start = b*num_views*num_frames
            mid = start+num_frames
            end = (b+1)*num_views*num_frames
            nn[start:mid] = mid+torch.argmin(distence[start:mid,mid:end], dim=1)
            nn[mid:end] = start+torch.argmin(distence[mid:end,start:mid], dim=1)
            if "single" in negative_type:
                weight[start:end,:start].fill_(0)
                weight[start:end,end:].fill_(0)
            if "noself" in negative_type:
                weight[start:mid,start:mid] = 0
                weight[mid:end,mid:end] = 0
        weight.masked_fill_((input_masks==0), 1e-6)

        # positive weight
        label = torch.zeros_like(logits)
        if positive_type == "gauss":
            pos_weight = torch.exp(-torch.square(distence)/(2*label_varience)).type_as(logits)
            # according to three sigma law, we can ignore the distence further than three sigma.
            # it may avoid the numerical unstablity and keep the performance.
            # pos_weight[(distence>3*np.sqrt(self.label_varience))] = 0
            for b in range(batch_size):
                start = b*num_views*num_frames
                mid = start+num_frames
                end = (b+1)*num_views*num_frames
                cur_pos_weight = pos_weight[start:mid,mid:end]
                label[start:mid,mid:end] = safe_div(cur_pos_weight, cur_pos_weight.sum(dim=1, keepdim=True))
                cur_pos_weight = pos_weight[mid:end,start:mid]
                label[mid:end,start:mid] = safe_div(cur_pos_weight, cur_pos_weight.sum(dim=1, keepdim=True))

        exp_logits = torch.exp(logits)
        sum_negative = torch.sum(weight*exp_logits, dim=1, keepdim=True)

        loss = F.kl_div(torch.log(safe_div(exp_logits, sum_negative) + 1e-6), label, reduction="none")
        loss = torch.sum(loss*input_masks)
        loss = loss / torch.sum(masks)
        
        return loss
