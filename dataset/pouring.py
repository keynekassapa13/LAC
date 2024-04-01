import torch
import os
import glob
import pickle
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from .util import create_data_augment, pad_zeros, read_video

class Pouring(torch.utils.data.Dataset):
    def __init__(self, cfg, pkl_name="train.pkl", mode="train"):
        self.cfg = cfg
        self.pkl_name = pkl_name
        self.mode = mode

        self.video_filenames = sorted(glob.glob(os.path.join(cfg.data_loader.data_dir + '/videos', "*.mp4")))

        with open(os.path.join(cfg.data_loader.data_dir, pkl_name), "rb") as f:
            self.dataset = pickle.load(f)

        self.max_seq_len = max([d["seq_len"] for d in self.dataset])
        
        self.num_frames = cfg.data_loader.num_frames
        self.num_contexts = cfg.data_loader.num_contexts
        self.num_context_steps = cfg.data_loader.num_context_steps
        self.frame_stride = cfg.data_loader.frame_stride
        self.sampling = cfg.data_loader.sampling
        self.random_offset = cfg.data_loader.random_offset
        self.context_stride = cfg.data_loader.context_stride

        self.augment = create_data_augment(cfg, augment=True)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        video_idx = self.dataset[idx]
        video_filename = video_idx["name"]

        video_path = os.path.join(self.cfg.data_loader.data_dir, "videos", f'{video_filename}.mp4')
        frames = read_video(video_path)
        
        seq_len = video_idx["seq_len"]

        frames = torch.tensor(frames)
        frames = frames.permute(0, 3, 1, 2).float() / 255.0

        if self.mode == "eval":
            # FRAMES: N x C x W x H
            return {
                "frames": frames,
                "seq_lens": seq_len,
                "steps": torch.arange(0, seq_len),
                "masks": torch.ones(seq_len),
                "labels": video_idx["frame_label"]
            }

        if self.cfg.type == "TCN":
            frames = pad_zeros(frames, self.max_seq_len)
            F, C, W, H = frames.shape
            tcn_frames = torch.zeros(self.cfg.data_loader.tcn_multiply, 3, C, W, H)

            for i in range(self.cfg.data_loader.tcn_multiply):
                anchor_index = np.random.randint(0, seq_len)
                positive_index = anchor_index +\
                    np.random.randint(-self.cfg.data_loader.positive_range, self.cfg.data_loader.positive_range)
                positive_index = np.clip(positive_index, 0, seq_len-1)
                lower_bound = (0, max(0, anchor_index - self.cfg.data_loader.negative_range))
                upper_bound = (min(anchor_index + self.cfg.data_loader.negative_range, seq_len), seq_len)
                negative_indices = np.concatenate([np.arange(*lower_bound), np.arange(*upper_bound)])
                negative_index = np.random.choice(negative_indices)
                tcn_frames[i, 0, :, :, :] = torch.from_numpy(frames[anchor_index, :, :, :])
                tcn_frames[i, 1, :, :, :] = torch.from_numpy(frames[positive_index, :, :, :])
                tcn_frames[i, 2, :, :, :] = torch.from_numpy(frames[negative_index, :, :, :])

            return {
                "frames": tcn_frames,
                "seq_lens": torch.tensor([seq_len]),
                "steps": torch.tensor([0]),
                "masks": torch.tensor([1]),
            }
            
        a_steps, a_chosen_steps, a_vmask = self.sample_frames(
            seq_len = seq_len,
            num_frames = self.num_frames,
            pre_steps = None
        )
        a_frames = self.augment(frames[a_steps.long()])

        b_steps, b_chosen_steps, b_vmask = self.sample_frames(
            seq_len = seq_len,
            num_frames = self.num_frames,
            pre_steps = a_steps
        )
        b_frames = self.augment(frames[b_steps.long()])

        ab_frames = torch.stack([a_frames, b_frames], dim=0)
        ab_steps = torch.stack([a_chosen_steps, b_chosen_steps], dim=0)
        ab_seq_lens = torch.tensor([seq_len, seq_len], dtype=torch.float32)
        ab_masks = torch.stack([a_vmask, b_vmask], dim=0)

        return {
            "frames": ab_frames,
            "steps": ab_steps,
            "seq_lens": ab_seq_lens,
            "masks": ab_masks
        }
    
    def sample_frames(self, seq_len, num_frames, pre_steps=None):
        sampling = self.cfg.data_loader.sampling
        pre_offset = min(pre_steps) if pre_steps is not None else 0

        if sampling == "offset_uniform":
            if seq_len >= num_frames:
                steps = torch.randperm(seq_len)
                steps = torch.sort(steps[:num_frames])[0]
            else:
                steps = torch.arange(0, num_frames)
        elif sampling == "time_augment":
            num_valid = min(seq_len, num_frames)
            expand_ratio = np.random.uniform(low=1.0, high=self.cfg.data_loader.sampling_region)\
                if self.cfg.data_loader.sampling_region > 1 else 1.0

            block_size = math.ceil(expand_ratio*seq_len)
            if pre_steps is not None and self.cfg.data_loader.consistent_offset != 0:
                shift = int((1-self.cfg.data_loader.consistent_offset)*num_valid)
                offset = np.random.randint(low=max(0, min(seq_len-block_size, pre_offset-shift)), 
                                           high=max(1, min(seq_len-block_size+1, pre_offset+shift+1)))
            else:
                offset = np.random.randint(low=0, high=max(seq_len-block_size, 1))
            steps = offset + torch.randperm(block_size)[:num_valid]
            steps = torch.sort(steps)[0]
            if num_valid < num_frames:
                steps = F.pad(steps, (0, num_frames-num_valid), "constant", seq_len)
        else:
            raise NotImplementedError
        
        video_mask = torch.ones(num_frames)
        video_mask[steps < 0] = 0
        video_mask[steps >= seq_len] = 0
        chosen_steps = torch.clamp(steps.clone(), 0, seq_len - 1)
        if self.num_contexts == 1:
            steps = chosen_steps
        else:
            context_stride = self.cfg.data_loader.context_stride
            steps = steps.view(-1,1) + context_stride*torch.arange(-(self.num_contexts-1), 1).view(1,-1)
            steps = torch.clamp(steps.view(-1), 0, seq_len - 1)
        return steps, chosen_steps, video_mask
