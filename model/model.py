from copy import deepcopy
import math
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .model_ import TransformerEmbModel, MLPHead, Classifier
from .model_ import BaseModel, ConvEmbedder
from .model_ import SpatialSoftmax

# ==================================================================================================
"""
References:
[1] Sermanet, P., Lynch, C., Chebotar, Y., Hsu, J., Jang, E., Schaal, S., & Levine, S. (2018). 
Time-Contrastive Networks: Self-Supervised Learning from Video.
[2] Finn, C., Tan, X. Y., Duan, Y., Darrell, T., Levine, S., & Abbeel, P. (2016). 
Deep Spatial Autoencoders for Visuomotor Learning.
[3]: https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834
"""
# ==================================================================================================


class Inceptionv3_SpatialSoftmax(nn.Module):
    def __init__(self, cfg):
        super(Inceptionv3_SpatialSoftmax, self).__init__()
        self.cfg = cfg
        inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.base_model = nn.Sequential(*list(inception.children())[:10])
        self.conv_block = nn.Sequential(
            nn.Conv2d(288, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=5),
            nn.ReLU(),
        )
        
        self.spatial_softmax = SpatialSoftmax(17, 17, 16,temperature=None)
        self.embedding = nn.Linear(32, 32)

    def forward(self, x, num_context=1):
        N, T, C, W, H = x.shape
        x = x.reshape([-1, C, W, H])

        x = self.base_model(x)
        x = self.conv_block(x)
        x = self.spatial_softmax(x)
        x = self.embedding(x)
        x = x.view(N, T, -1)
        return x
    
# ==================================================================================================
"""
References:
[1] Dwibedi, D., Aytar, Y., Tompson, J., Sermanet, P., & Zisserman, A. (2019). 
Temporal Cycle-Consistency Learning. 
"""
# ==================================================================================================
    
class ResNet50_Conv(nn.Module):
    def __init__(self, cfg):
        super(ResNet50_Conv, self).__init__()
        self.cfg = cfg
        self.model = torch.hub.load('pytorch/vision', 'resnet50', weights='ResNet50_Weights.IMAGENET1K_V1')
        # self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.features = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3[0:3],
            # self.model.layer4[0:3]
        )
        
        # CONV
        self.conv_layers = [
            nn.Conv3d(in_channels=1024, out_channels=512, kernel_size=(3,3,3), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3,3,3), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(True)
        ]
        self.conv_layers = nn.Sequential(*self.conv_layers)
        
        # POOL
        self.pooling = nn.AdaptiveMaxPool3d(1)
        
        # FCN
        self.fc_layers = [
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(True)
        ]
        self.fc_layers = nn.Sequential(*self.fc_layers)
        
        # EMBED
        self.embedding_layer = nn.Linear(512, 128)

    def forward(self, x, num_context=2):
        
        N, T, C, W, H = x.shape
        x = x.reshape([-1, C, W, H])
        
        x = self.features(x)
        _, c, w, h = x.shape
        x = x.reshape([N, T, c, w, h])
        # x = x.permute(0, 2, 1, 3, 4) # N C T W H
        
        num_frames = T // num_context
        x = x.view(N * num_frames, num_context, c, h, w)
        x = x.transpose(1,2)
        
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        x = self.embedding_layer(x)
        x = x.view(N, num_frames, 128)
        return x
    
# ==================================================================================================
"""
References:
[1] Hadji, I., Derpanis, K. G., & Jepson, A. D. (2021). 
Representation Learning via Global Temporal Alignment and Cycle-Consistency.
[2] https://github.com/hadjisma/VideoAlignment/tree/master for GTA loss
[3] https://github.com/trquhuytin/LAV-CVPR21 for base model and conv embedder
"""
# ==================================================================================================
    
class ResNet50_Conv2(nn.Module):
    def __init__(self, cfg):
        super(ResNet50_Conv2, self).__init__()

        self.cfg = cfg

        self.base_cnn = BaseModel(pretrained=True)
        self.emb = ConvEmbedder(emb_size=cfg.arch.embedding_size, 
                                l2_normalize=True)

    def forward(self, x, num_context=None):
        num_ctxt = self.cfg.data_loader.num_context_steps
        if num_context:
            num_ctxt = num_context

        num_frames = x.size(1) // num_ctxt
        x = self.base_cnn(x)
        x = self.emb(x, num_frames)
        return x
    
# ==================================================================================================
"""
References:
[1] Chen, M., Wei, F., Li, C., & Cai, D. (2022). 
Frame-wise Action Representations for Long Videos via Sequence Contrastive Learning. 
[2] https://github.com/minghchen/CARL_code
"""
# ==================================================================================================

class ResNet50_Transformer1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        res50_model = models.resnet50()
        
        pretrained_weights = "./pretrained_models/BYOL_1000.pth"
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        state_dict = {}
        for key, value in checkpoint["model"].items():
            if 'encoder_k' in key: continue
            if 'encoder' in key:
                new_key = key.replace('module.encoder.', '')
                state_dict[new_key] = value
        msg = res50_model.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"Load pretrained weights from {pretrained_weights}")

        if cfg.arch.base_model.layer == 3:
            self.backbone = nn.Sequential(*list(res50_model.children())[:-3]) # output of layer3: 1024x14x14
            self.res_finetune = list(res50_model.children())[-3]
            cfg.arch.base_model.out_channel = 2048
        elif cfg.arch.base_model.layer == 2:
            self.backbone = nn.Sequential(*list(res50_model.children())[:-4]) # output of layer2
            self.res_finetune = nn.Sequential(*list(res50_model.children())[-4:-2])
            cfg.arch.base_model.out_channel = 2048
        else:
            self.backbone = nn.Sequential(*list(res50_model.children())[:-2]) # output of layer4: 2048x7x7
            self.res_finetune = nn.Identity()
            cfg.arch.base_model.out_channel = 2048
        self.embed = TransformerEmbModel(cfg)
        self.embedding_size = self.embed.embedding_size
        
        if cfg.arch.projection:
            self.ssl_projection = MLPHead(cfg)
        if cfg.arch.type == 'classification':
            self.classifier = Classifier(cfg)

    def forward(self, x, num_frames=None, video_masks=None, project=False, classification=False):

        batch_size, num_steps, c, h, w = x.shape
        frames_per_batch = self.cfg.arch.base_model.frames_per_batch
        num_blocks = int(math.ceil(float(num_steps)/frames_per_batch))
        backbone_out = []
        for i in range(num_blocks):
            curr_idx = i * frames_per_batch
            cur_steps = min(num_steps-curr_idx, frames_per_batch)
            curr_data = x[:, curr_idx:curr_idx+cur_steps]
            curr_data = curr_data.contiguous().view(-1, c, h, w)
            self.backbone.eval()
            with torch.no_grad():
                curr_emb = self.backbone(curr_data)
            curr_emb = self.res_finetune(curr_emb)
            _, out_c, out_h, out_w = curr_emb.size()
            curr_emb = curr_emb.contiguous().view(batch_size, cur_steps, out_c, out_h, out_w)
            backbone_out.append(curr_emb)
        x = torch.cat(backbone_out, dim=1)
        
        x = self.embed(x, video_masks=video_masks)

        if self.cfg.arch.projection and project:
            x = self.ssl_projection(x)
            x = F.normalize(x, dim=-1)
        elif self.cfg.arch.l2_normalize:
            x = F.normalize(x, dim=-1)
        if classification:
            return self.classifier(x)
        return x


# ==================================================================================================
"""
LAC
"""
# ==================================================================================================

def generate_sincos_embedding(seq_len, d_model, train_len=None):
    odds = np.arange(0, d_model, 2)
    evens = np.arange(1, d_model, 2)
    pos_enc_mat = np.zeros((seq_len, d_model))
    if train_len is None:
        pos_list = np.arange(seq_len)
    else:
        pos_list = np.linspace(0, train_len-1, num=seq_len)

    for i, pos in enumerate(pos_list):
        pos_enc_mat[i, odds] = np.sin(pos / (10000 ** (odds / d_model)))
        pos_enc_mat[i, evens] = np.cos(pos / (10000 ** (evens / d_model)))

    return torch.from_numpy(pos_enc_mat).unsqueeze(0)

class PositionalEncoder(nn.Module):
    def __init__(self, cfg, d_model, dout_p, seq_len=3660):
        super(PositionalEncoder, self).__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)
        self.seq_len = seq_len

    def forward(self, x):
        B, S, d_model = x.shape
        if S != self.seq_len:
            pos_enc_mat = generate_sincos_embedding(S, d_model, self.seq_len)
            x = x + pos_enc_mat.type_as(x)
        else:
            pos_enc_mat = generate_sincos_embedding(S, d_model)
            x = x + pos_enc_mat.type_as(x)
        x = self.dropout(x)
        return x

class ResNet50_Transformer2(nn.Module):
    def __init__(self, cfg):
        super(ResNet50_Transformer2, self).__init__()
        self.cfg = cfg
        self.model = torch.hub.load('pytorch/vision', 'resnet50', weights='ResNet50_Weights.IMAGENET1K_V2')
        # self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.features = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3[0:3],
            # self.model.layer4[0:3]
        )
        
        # CONV
        self.conv_layers = [
            nn.Conv3d(in_channels=1024, out_channels=512, kernel_size=(3,3,3), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3,3,3), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(True)
        ]
        self.conv_layers = nn.Sequential(*self.conv_layers)
        
        # POOL
        self.pooling = nn.AdaptiveMaxPool3d(1)
        
        # FCN
        self.fc_layers = [
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(True)
        ]
        self.fc_layers = nn.Sequential(*self.fc_layers)
        
        # EMBED
        self.embedding_layer = nn.Linear(512, 256)

        # POS_ENC
        self.pos_enc = PositionalEncoder(cfg, 256, 0.1, seq_len=32)

        # TRANSFORMER
        encoder_layers = TransformerEncoderLayer(256, 8, 1024, 0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)

        # EMBED 2
        self.embedding_layer2 = nn.Linear(256, 128)


    def forward(self, x, num_context=2):
        
        N, T, C, W, H = x.shape
        x = x.reshape([-1, C, W, H])
        
        x = self.features(x)
        _, c, w, h = x.shape
        x = x.reshape([N, T, c, w, h])
        # x = x.permute(0, 2, 1, 3, 4) # N C T W H
        
        num_frames = T // num_context
        x = x.view(N * num_frames, num_context, c, h, w)
        x = x.transpose(1,2)
        
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        x = self.embedding_layer(x)
        # x = x.view(N, num_frames, 128)
        x = x.view(N, num_frames, -1)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        x = x.view(N * num_frames, -1)
        x = self.embedding_layer2(x)
        x = x.view(N, num_frames, -1)
        
        return x