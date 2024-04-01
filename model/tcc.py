import torch
import torch.nn.functional as F

from loguru import logger
from .mod import MOD

"""
References:
[1] Dwibedi, D., Aytar, Y., Tompson, J., Sermanet, P., & Zisserman, A. (2019). 
Temporal Cycle-Consistency Learning.
[2] https://github.com/google-research/google-research/tree/master/tcc
"""

class TCC(MOD):
    def __init__(self, cfg):
        self.cfg = cfg
        self.criterion = self.create_criterion()
        self.shape = ()

    def create_criterion(self):
        if self.cfg.loss.type == "stochastic":
            loss = self.stochastic_loss
        else:
            loss = self.deterministic_loss
        return loss
    
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
        steps = steps.reshape(self.shape[0] * self.shape[1], -1)
        return steps
    
    def adjust_seq_lens(self, seq_lens):
        seq_lens = seq_lens.reshape(self.shape[0] * self.shape[1])
        return seq_lens
    
    def adjust_embed(self, embed):
        return embed
    
    def calculate_loss(self, embed, steps, seq_lens, masks=None):
        embed = self.adjust_embed(embed)
        loss = self.criterion(
            embed, steps, seq_lens,
            num_steps=self.cfg.data_loader.num_steps,
            batch_size=self.cfg.data_loader.batch_size,
            loss_type=self.cfg.loss.loss_type,
            similarity_type=self.cfg.loss.similarity_type,
            num_cycles=5,
            cycle_length=2,
            temperature=self.cfg.loss.temperature,
            label_smoothing=self.cfg.loss.label_smoothing,
            variance_lambda=self.cfg.loss.variance_lambda,
            huber_delta=self.cfg.loss.huber_delta,
            normalize_indices=self.cfg.loss.normalize_indices
        )
        return loss
    
    def classification_loss(self, logits, labels, label_smoothing):
        labels = labels.detach()
        return torch.mean(F.cross_entropy(logits, labels, reduction='mean', label_smoothing=label_smoothing))

    def regression_loss(
        self, logits, labels, 
        num_steps, steps, 
        seq_lens, loss_type, 
        normalize_indices, 
        variance_lambda, huber_delta):
        
        steps = steps.cuda()
        labels = labels.cuda()
        
        steps = steps.detach()
        labels = labels.detach()
        
        float_seq_lens = seq_lens.float()
        tile_seq_lens = float_seq_lens.unsqueeze(1).expand(-1, num_steps)
        steps = steps.float() / tile_seq_lens
        
        beta = torch.softmax(logits, dim=-1)
        true_time = torch.sum(steps * labels, dim=-1)
        pred_time = torch.sum(steps * beta, dim=-1)

        if loss_type in ['regression_mse', 'regression_mse_var']:
            if 'var' in loss_type:
                pred_time_variance = torch.sum(torch.square(steps - pred_time.unsqueeze(-1)) * beta, dim=-1)
                pred_time_log_var = torch.log(pred_time_variance)
                squared_error = torch.square(true_time - pred_time)
                return torch.mean(torch.exp(-pred_time_log_var) * squared_error
                                            + variance_lambda * pred_time_log_var)
            else:
                return torch.nn.MSELoss(reduction="mean")(pred_time, true_time)
        elif loss_type == 'regression_huber':
            return torch.mean(F.huber_loss(true_time, pred_time, delta=huber_delta, reduction='mean'))
        

    def _align_single_cycle(self, cycle, embs, cycle_length, num_steps, similarity_type, temperature):
        n_idx = torch.randint(0, num_steps - 1, (1,))
        onehot_labels = F.one_hot(n_idx, num_steps).float()
        
        query_feats = embs[cycle[0], n_idx:n_idx+1]
        num_channels = query_feats.size(-1)
        
        for c in range(1, cycle_length + 1):
            candidate_feats = embs[cycle[c]]
            if similarity_type == 'l2':
                tmp = torch.square(query_feats - candidate_feats)
                mean_squared_distance = tmp.sum(1)
                similarity = - mean_squared_distance
            elif similarity_type == 'cosine':
                similarity = torch.matmul(candidate_feats, query_feats.T).squeeze()
            else:
                raise ValueError('similarity_type can either be l2 or cosine.')
        
            similarity /= float(num_channels)
            similarity /= temperature
            
            beta = torch.softmax(similarity, dim=-1)
            beta = beta.unsqueeze(-1).expand(-1, num_channels)
            
            query_feats = torch.sum(beta * candidate_feats, dim=0, keepdim=True)
        return similarity, onehot_labels

    def _align(self, cycles, embs, num_steps, num_cycles, cycle_length, similarity_type, temperature):
        logits_list = []
        labels_list = []
        for i in range(num_cycles):
            logits, labels = self._align_single_cycle(
                cycles[i], embs, cycle_length, num_steps, similarity_type, temperature
            )
            logits_list.append(logits)
            labels_list.append(labels)
            
        logits = torch.stack(logits_list)
        labels = torch.stack(labels_list)
        
        return logits, labels

    def gen_cycles(self, num_cycles, batch_size, cycle_length=2):
        range_tensor = torch.arange(batch_size)
        sorted_idxes = range_tensor.unsqueeze(0).expand(num_cycles, -1)
        sorted_idxes = sorted_idxes.reshape([batch_size, num_cycles])
        cycles = sorted_idxes[torch.randperm(len(sorted_idxes))].reshape([num_cycles, batch_size])
        cycles = cycles[:, :cycle_length]
        cycles = torch.cat([cycles, cycles[:, 0:1]], dim=-1)
        return cycles
    
    def stochastic_loss(
            self, embs, steps, seq_lens, num_steps,
            batch_size, loss_type="regression_mse_var", similarity_type="l2",
            num_cycles=5, cycle_length=2, temperature=0.1,
            label_smoothing=0.0, variance_lambda=0.001, huber_delta=1.0,
            normalize_indices=True
    ):
        cycles = self.gen_cycles(num_cycles, batch_size, cycle_length)
        logits, labels = self._align(cycles, embs, num_steps, num_cycles, cycle_length, similarity_type, temperature)

        if loss_type == 'classification':
            loss = self.classification_loss(logits, labels, label_smoothing)
        elif 'regression' in loss_type:
            cycles_indices = cycles[:, 0]
            steps = torch.stack([steps[index, :] for index in cycles_indices])
            seq_lens = torch.stack([seq_lens[index] for index in cycles_indices])
            loss = self.regression_loss(logits, labels, num_steps, steps, seq_lens, loss_type, normalize_indices, variance_lambda, huber_delta)
        else:
            raise ValueError(f'Unidentified loss type {loss_type}. Currently supported loss types are: regression_mse, regression_huber, classification.')
        return loss
    

    def pairwise_l2_distance(self, embs1, embs2):
        norm1 = torch.square(embs1).sum(1).view(-1,1)
        norm2 = torch.square(embs2).sum(1).view(1,-1)
        dist = - torch.max(norm1 + norm2 - 2*torch.mm(embs1, embs2.T), torch.tensor(0))
        return dist

    def get_scaled_similarity(self, embs1, embs2, similarity_type, temperature):
        num_steps, channels = embs1.shape
        
        if similarity_type == 'cosine':
            similarity = torch.matmul(embs1, embs2.T)
        elif similarity_type == 'l2':
            similarity = self.pairwise_l2_distance(embs1, embs2)
        similarity /= channels
        similarity /= temperature
        return similarity

    def align_pair_of_sequences(self, embs1, embs2, similarity_type, temperature):
        num_steps, channels = embs1.shape
        
        sim_12 = self.get_scaled_similarity(embs1, embs2, similarity_type, temperature)
        
        softmaxed_sim_12 = torch.softmax(sim_12, dim=-1)
        nn_embs = torch.matmul(softmaxed_sim_12, embs2)
        
        sim_21 = self.get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)
        
        logits = sim_21
        # labels = torch.diag(torch.ones(num_steps)).type_as(logits)
        labels = F.one_hot(torch.arange(num_steps), num_classes=num_steps)
        labels = labels.type_as(logits)
        
        return logits, labels
    
    def deterministic_loss(
            self, embs, steps, seq_lens, num_steps,
            batch_size, loss_type="regression_mse", similarity_type="l2",
            num_cycles=5, cycle_length=2, temperature=0.1,
            label_smoothing=0.1, variance_lambda=0.001, huber_delta=1.0,
            normalize_indices=True
    ):      
            
        labels_list = []
        logits_list = []
        steps_list = []
        seq_lens_list = []
        
        batch_size, num_frames, channels = embs.shape

        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    logits, labels = self.align_pair_of_sequences(embs[i], embs[j], similarity_type, temperature)
                    logits_list.append(logits)
                    labels_list.append(labels)
                    steps_list.append(steps[i].unsqueeze(0).expand(num_frames, num_frames))
                    seq_lens_list.append(seq_lens[i].view(1,).expand(num_frames))

        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        steps = torch.cat(steps_list, dim=0)
        seq_lens = torch.cat(seq_lens_list, dim=0)

        if loss_type == 'classification':
            loss = self.classification_loss(logits, labels, label_smoothing)
        elif 'regression' in loss_type:
            loss = self.regression_loss(
                logits, labels, num_steps, 
                steps, seq_lens, loss_type, 
                normalize_indices, variance_lambda, huber_delta)
        return loss