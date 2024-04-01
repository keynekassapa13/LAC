import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from loguru import logger
from datetime import datetime
from tqdm import tqdm
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from evaluation import get_tasks

from .model import *

from .tcn import TCN
from .tcc import TCC
from .lav import LAV
from .gta import GTA
from .scl import SCL
from .lac import LAC

align_dict = {
    "TCN": TCN,
    "TCC": TCC,
    "LAV": LAV,
    "GTA": GTA,
    "SCL": SCL,
    "LAC": LAC,
}

model_dict = {
    "Inceptionv3_SpatialSoftmax": Inceptionv3_SpatialSoftmax,
    "ResNet50_Conv": ResNet50_Conv,
    "ResNet50_Conv2": ResNet50_Conv2,
    "ResNet50_Transformer1": ResNet50_Transformer1,
    "ResNet50_Transformer2": ResNet50_Transformer2,
}

def load_ckpt(cfg, model, optimizer):
    path = cfg.trainer.save_dir
    if os.path.exists(path):
        model_names = [m for m in os.listdir(path) if m.endswith(".pth")]
        if len(model_names) > 0:
            model_names.sort()
            model_name = model_names[-1]
            path = os.path.join(path, model_name)
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from {path}")
            return model, optimizer, checkpoint["epoch"]
        
    return model, optimizer, 0

def save_ckpt(cfg, model, optimizer, epoch):
    path = os.path.join(cfg.trainer.save_dir, f"{cfg.type}-ckpt_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, path)
    logger.info(f"Saved checkpoint to {path}")

def train(cfg, train_loader, train_eval_loader=None, val_eval_loader=None):
    model = model_dict[cfg.arch.type](cfg)
    model_cfg = align_dict[cfg.type](cfg)

    if cfg.trainer.log_summary:
        summary(model, input_size=(2, 64, 3, 224, 224))
    
    model.to(cfg.device)
    model.train()
    
    optimizer = model_cfg.create_optimizer(model.parameters())
    
    last_epoch = 0
    if cfg.trainer.resume:
        model, optimizer, last_epoch = load_ckpt(cfg, model, optimizer)

    logger.info(f"Logging to {cfg.trainer.log_dir}")
    writer = SummaryWriter(log_dir=f"{cfg.trainer.log_dir}/{cfg.name}-{datetime.now().strftime('%Y-%m-%d')}")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=cfg.trainer.epochs+1, 
        eta_min=0, last_epoch=-1)
    
    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Using AMP")

    for epoch in tqdm(range(last_epoch, last_epoch+cfg.trainer.epochs)):
        avg_loss = 0
        
        for i, data in enumerate(train_loader):
            model.train()

            frames = data["frames"].to(cfg.device)
            seq_lens = data["seq_lens"].to(cfg.device)
            steps = data["steps"].to(cfg.device)
            masks = data["masks"].to(cfg.device)

            frames = model_cfg.adjust_frames(frames)
            steps = model_cfg.adjust_steps(steps)
            seq_lens = model_cfg.adjust_seq_lens(seq_lens)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                if cfg.arch.type == "ResNet50_Transformer1":
                    embed = model(
                        frames, 
                        video_masks=masks.view(-1, 1, cfg.data_loader.num_frames),
                        project=True)
                else:
                    embed = model(frames, num_context=cfg.data_loader.num_contexts)
            loss = model_cfg.calculate_loss(embed, steps, seq_lens, masks)
            
            if cfg.use_amp:
                scaler.scale(loss).backward()
                if cfg.optimizer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.optimizer.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
                optimizer.step()
            
            avg_loss += loss.item()

            if cfg.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            logger.info(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

        avg_loss /= len(train_loader)
        scheduler.step()

        logger.info(f"Epoch: {epoch}, Loss: {avg_loss}")

        writer.add_scalar('train/lr', [param_group["lr"] for param_group in optimizer.param_groups][0], epoch)
        writer.add_scalar('train/loss', avg_loss, epoch)

        if epoch % cfg.trainer.log_interval == 0:
            save_ckpt(cfg, model, optimizer, epoch)

        if cfg.eval:
            eval(cfg, model, model_cfg, train_eval_loader, val_eval_loader, epoch, writer)
    
    writer.close()
    save_ckpt(cfg, model, optimizer, epoch)
    logger.info(f"Training completed. Model saved to {cfg.trainer.save_dir}")

    return

def get_embed_eval(cfg, model, loader):
    model.eval()

    embs_list = []
    steps_list = []
    seq_lens_list = []
    labels_list = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            frames = data["frames"].to(cfg.device)
            seq_lens = data["seq_lens"]
            steps = data["steps"]
            labels = data["labels"]
            masks = data["masks"]

            if cfg.arch.type == "ResNet50_Transformer1":
                embed = model(
                    frames, 
                    project=True)
            else:
                embed = model(frames, num_context=1)
            
            valid = (labels[0] >= 0)

            embs_list.append(embed[0][valid].cpu().numpy())
            seq_lens_list.append(seq_lens)
            labels_list.append(labels[0][valid])
    
    return {
        "embs": embs_list,
        "seq_lens": seq_lens_list,
        "labels": labels_list,
    }
    
def eval(cfg, model, model_cfg, train_eval_loader, val_eval_loader, epoch, summary_writer):
    if cfg.data_loader.type.lower() == "pouring":
        dataset = {"name": "pouring"}
    elif cfg.data_loader.type.lower() == "pennaction":
        dataset = {"name": cfg.data_loader.action}
    else:
        raise ValueError("Invalid dataset type")
    
    tasks = get_tasks(cfg)

    with torch.no_grad():
        dataset["train_dataset"] = get_embed_eval(cfg, model, train_eval_loader)
        dataset["val_dataset"] = get_embed_eval(cfg, model, val_eval_loader)

        metrics = {}
        for task_name, task in tasks.items():
            metrics[task_name] = task.evaluate(dataset, epoch, summary_writer)
            logger.info(f"{task_name} done!")
        
        for task_name in tasks.keys():
            summary_writer.add_scalar('metrics/%s_%s' % ("pouring", task_name),
                                    metrics[task_name], epoch)

        del dataset
        return