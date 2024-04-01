import torch

from .mod import MOD

"""
References:
    [1]: Finn, C., Tan, X. Y., Duan, Y., Darrell, T., Levine, S., & Abbeel, P. (2016). 
    Deep Spatial Autoencoders for Visuomotor Learning.
    [2]: https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834
"""

class TCN(MOD):
    def __init__(self, cfg):
        self.cfg = cfg
        self.criterion = self.create_criterion()
        self.shape = ()

    def create_criterion(self):
        criterion = torch.nn.TripletMarginLoss(
            margin=self.cfg.loss.margin, 
            p=self.cfg.loss.pairwise_distance, 
            reduction=self.cfg.loss.reduction)
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

        frames = frames.reshape(N, -1, C, W, H)
        return frames
    
    def adjust_embed(self, embed):
        embed = embed.reshape(self.shape[0] * self.shape[1], self.shape[2], -1)
        return embed
    
    def calculate_loss(self, embed, steps, seq_lens, masks=None):
        embed = self.adjust_embed(embed)
        loss = self.criterion(
            embed[:, 0, :],
            embed[:, 1, :],
            embed[:, 2, :]
        )
        return loss