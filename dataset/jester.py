import os
import glob
import math
import numpy as np
import torch
import torch.nn.functional as F

from loguru import logger
from .util import create_data_augment, read_video


class Jester(torch.utils.data.Dataset):
    """20BN-Jester loader for self-supervised alignment.

    Unlike Pouring/PennAction there is no train.pkl/val.pkl: each clip is just an
    mp4 under <data_dir>/videos, and two temporally-augmented views are sampled
    per clip. The `mode` argument is accepted for a uniform construct() interface
    but Jester only supports SSL training (cfg.eval should be false).
    """

    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.data_dir = cfg.data_loader.data_dir

        self.video_filenames = sorted(
            glob.glob(os.path.join(cfg.data_loader.data_dir + "/videos", "*.mp4"))
        )
        self.dataset = self.video_filenames
        if len(self.dataset) == 0:
            logger.warning(f"No mp4 videos found under {cfg.data_loader.data_dir}/videos")

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
        frames = read_video(self.dataset[idx])
        seq_len = frames.shape[0]

        frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0

        a_steps, a_chosen_steps, a_vmask = self.sample_frames(seq_len, self.num_frames, pre_steps=None)
        a_frames = self.augment(frames[a_steps.long()])

        b_steps, b_chosen_steps, b_vmask = self.sample_frames(seq_len, self.num_frames, pre_steps=a_steps)
        b_frames = self.augment(frames[b_steps.long()])

        return {
            "frames": torch.stack([a_frames, b_frames], dim=0),
            "steps": torch.stack([a_chosen_steps, b_chosen_steps], dim=0),
            "seq_lens": torch.tensor([seq_len, seq_len], dtype=torch.float32),
            "masks": torch.stack([a_vmask, b_vmask], dim=0),
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
            expand_ratio = np.random.uniform(low=1.0, high=self.cfg.data_loader.sampling_region) \
                if self.cfg.data_loader.sampling_region > 1 else 1.0

            block_size = math.ceil(expand_ratio * seq_len)
            if pre_steps is not None and self.cfg.data_loader.consistent_offset != 0:
                shift = int((1 - self.cfg.data_loader.consistent_offset) * num_valid)
                offset = np.random.randint(low=max(0, min(seq_len - block_size, pre_offset - shift)),
                                           high=max(1, min(seq_len - block_size + 1, pre_offset + shift + 1)))
            else:
                offset = np.random.randint(low=0, high=max(seq_len - block_size, 1))
            steps = offset + torch.randperm(block_size)[:num_valid]
            steps = torch.sort(steps)[0]
            if num_valid < num_frames:
                steps = F.pad(steps, (0, num_frames - num_valid), "constant", seq_len)
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
            steps = steps.view(-1, 1) + context_stride * torch.arange(-(self.num_contexts - 1), 1).view(1, -1)
            steps = torch.clamp(steps.view(-1), 0, seq_len - 1)
        return steps, chosen_steps, video_mask
