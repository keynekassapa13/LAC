class MOD:
    def __init__(self, cfg):
        self.cfg = cfg
        self.criterion = self.create_criterion()
        self.shape = ()

    def create_criterion(self):
        return None
    
    def create_optimizer(self, model_param):
        return None
    
    def adjust_frames(self, frames):
        return frames
    
    def adjust_steps(self, steps):
        return steps
    
    def adjust_seq_lens(self, seq_lens):
        return seq_lens
    
    def adjust_embed(self, embed):
        return embed
    
    def calculate_loss(self, embed, steps, seq_lens, masks=None):
        return 0