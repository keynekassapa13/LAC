import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights


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

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel, temperature=None):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel
        
        if temperature:
            self.temperature = torch.ones(1) * temperature
        else:
            self.temperature = nn.Parameter(torch.ones(1))
        
        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()

        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        feature = feature.reshape(-1, self.height * self.width)

        softmax_attention = nn.functional.softmax(feature / self.temperature, dim=-1)
        
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

# ==================================================================================================
"""
References:
[1] Hadji, I., Derpanis, K. G., & Jepson, A. D. (2021). 
Representation Learning via Global Temporal Alignment and Cycle-Consistency.
[2] https://github.com/hadjisma/VideoAlignment/tree/master for GTA loss
[3] https://github.com/trquhuytin/LAV-CVPR21 for base model and conv embedder
"""
# ==================================================================================================
    
class BaseModel(nn.Module):

    def __init__(self, pretrained=True):
        super(BaseModel, self).__init__()
        
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        layers = list(resnet.children())[:-3]
        layers[-1] = nn.Sequential(*list(layers[-1].children())[:-3])
        self.base_model = nn.Sequential(*layers)

    def forward(self, x):

        batch_size, num_steps, c, h, w = x.shape
        x = torch.reshape(x, [batch_size * num_steps, c, h, w])

        x = self.base_model(x)

        _, c, h, w = x.shape
        x = torch.reshape(x, [batch_size, num_steps, c, h, w])

        return x
    
class ConvEmbedder(nn.Module):

    def __init__(self, emb_size=128, l2_normalize=False):
        super(ConvEmbedder, self).__init__()

        self.emb_size = emb_size
        self.l2_normalize = l2_normalize

        self.conv1 = nn.Conv3d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(512)

        self.conv2 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.1)

        self.embedding_layer = nn.Linear(512, emb_size)
    
    def apply_bn(self, bn, x):
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = torch.reshape(x, (-1, x.shape[-1]))
        x = bn(x)
        x = torch.reshape(x, (N, T, H, W, C))
        x = x.permute(0, 4, 1, 2, 3)
        return x

    def forward(self, x, num_frames):

        batch_size, total_num_steps, c, h, w = x.shape
        num_context = total_num_steps // num_frames
        x = torch.reshape(x, (batch_size * num_frames, num_context, c, h, w))

        x = x.transpose(1, 2)

        x = self.conv1(x)

        x = self.apply_bn(self.bn1, x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.apply_bn(self.bn2, x)
        x = F.relu(x)

        x = torch.max(x.view(x.size(0), x.size(1), -1), dim=-1)[0]
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.embedding_layer(x)

        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=-1)
        
        x = torch.reshape(x, (batch_size, num_frames, self.emb_size))
        return x

# ==================================================================================================
"""
References:
[1] Chen, M., Wei, F., Li, C., & Cai, D. (2022). 
Frame-wise Action Representations for Long Videos via Sequence Contrastive Learning. 
[2] https://github.com/minghchen/CARL_code
"""
# ==================================================================================================


def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class ResidualConnection(nn.Module):
    def __init__(self, size, dout_p):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)

    def forward(self, x, sublayer): 
        # x (B, S, D)
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)

        return x + res
    
def attention(Q, K, V, mask=None, dropout=None, visual=False):
    # Q, K, V are (B, *(H), seq_len, d_model//H = d_k)
    # mask is     (B,    1,       1,               Ss)
    d_k = Q.size(-1)
    # (B, H, S, S)
    QKt = Q.matmul(K.transpose(-1, -2))
    sm_input = QKt / np.sqrt(d_k)

    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))

    softmax = F.softmax(sm_input, dim=-1)
    out = softmax.matmul(V)

    if dropout is not None:
        out = dropout(out)

    # (B, *(H), seq_len, d_model//H = d_k)
    if visual:
        return out, softmax.detach()
    else:
        return out
    
class MultiheadedAttention(nn.Module):
    def __init__(self, d_model_Q, d_model_K, d_model_V, H, dout_p=0.0, d_model=None, d_out=None):
        super(MultiheadedAttention, self).__init__()
        self.d_model_Q = d_model_Q
        self.d_model_K = d_model_K
        self.d_model_V = d_model_V
        self.H = H
        self.d_model = d_model
        self.dout_p = dout_p
        self.d_out = d_out
        if self.d_out is None:
            self.d_out = self.d_model_Q

        if self.d_model is None:
            self.d_model = self.d_model_Q

        self.d_k = self.d_model // H

        self.linear_Q2d = nn.Linear(self.d_model_Q, self.d_model)
        self.linear_K2d = nn.Linear(self.d_model_K, self.d_model)
        self.linear_V2d = nn.Linear(self.d_model_V, self.d_model)
        self.linear_d2Q = nn.Linear(self.d_model, self.d_out)

        self.dropout = nn.Dropout(self.dout_p)
        self.visual = False

        assert self.d_model % H == 0

    def forward(self, Q, K, V, mask=None):
        ''' 
            Q, K, V: (B, Sq, Dq), (B, Sk, Dk), (B, Sv, Dv)
            mask: (B, 1, Sk)
            Sk = Sv, 
            Dk != self.d_k
        '''
        B, Sq, d_model_Q = Q.shape
        # (B, Sm, D) <- (B, Sm, Dm)
        Q = self.linear_Q2d(Q)
        K = self.linear_K2d(K)
        V = self.linear_V2d(V)

        # (B, H, Sm, d_k) <- (B, Sm, D)
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2)  # (-4, -3*, -2*, -1)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)

        if mask is not None:
            # the same mask for all heads -> (B, 1, 1, Sm2)
            mask = mask.unsqueeze(1)

        # (B, H, Sq, d_k) <- (B, H, Sq, d_k), (B, H, Sk, d_k), (B, H, Sv, d_k), Sk = Sv
        if self.visual:
            Q, self.attn_matrix = attention(Q, K, V, mask, self.dropout, self.visual)
            self.attn_matrix = self.attn_matrix.mean(-3)
        else:
            Q = attention(Q, K, V, mask, self.dropout)
        # (B, Sq, D) <- (B, H, Sq, d_k)
        Q = Q.transpose(-3, -2).contiguous().view(B, Sq, self.d_model)
        # (B, Sq, Dq)
        Q = self.linear_d2Q(Q)

        return Q

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dout_p):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dout_p = dout_p
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dout_p)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        '''In, Out: (B, S, D)'''
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, dout_p, H=8, d_ff=None, d_hidden=None):
        super(EncoderLayer, self).__init__()
        self.res_layer0 = ResidualConnection(d_model, dout_p)
        self.res_layer1 = ResidualConnection(d_model, dout_p)
        if d_hidden is None: d_hidden = d_model
        if d_ff is None: d_ff = 4*d_model
        self.self_att = MultiheadedAttention(d_model, d_model, d_model, H, d_model=d_hidden)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dout_p=0.0)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x, src_mask=None):
        '''
        in:
            x: (B, S, d_model), src_mask: (B, 1, S)
        out:
            (B, S, d_model)
        '''
        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs 
        # the output of the self attention
        sublayer0 = lambda x: self.self_att(x, x, x, src_mask)
        sublayer1 = self.feed_forward
        
        x = self.res_layer0(x, sublayer0)
        x = self.res_layer1(x, sublayer1)
        
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, dout_p, H, d_ff, N, d_hidden=None):
        super(Encoder, self).__init__()
        self.enc_layers = clone(EncoderLayer(d_model, dout_p, H, d_ff, d_hidden), N)
        
    def forward(self, x, src_mask=None):
        '''
        in:
            x: (B, S, d_model) src_mask: (B, 1, S)
        out:
            # x: (B, S, d_model) which will be used as Q and K in decoder
        '''
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x
    
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

class TransformerEmbModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        drop_rate = cfg.arch.embedder_model.dropout
        in_channels = cfg.arch.base_model.out_channel
        cap_scalar = cfg.arch.embedder_model.capacity_scalar
        fc_params = cfg.arch.embedder_model.fc_layers
        self.embedding_size = cfg.arch.embedder_model.embedding_size
        hidden_channels = cfg.arch.embedder_model.hidden_size
        self.pooling = nn.AdaptiveMaxPool2d(1)
        
        self.fc_layers = []
        for channels, activate in fc_params:
            channels = channels*cap_scalar
            self.fc_layers.append(nn.Dropout(drop_rate))
            self.fc_layers.append(nn.Linear(in_channels, channels))
            self.fc_layers.append(nn.BatchNorm1d(channels))
            self.fc_layers.append(nn.ReLU(True))
            in_channels = channels
        self.fc_layers = nn.Sequential(*self.fc_layers)
        
        self.video_emb = nn.Linear(in_channels, hidden_channels)
        
        self.video_pos_enc = PositionalEncoder(cfg, hidden_channels, drop_rate, seq_len=cfg.data_loader.num_steps)
        if cfg.arch.embedder_model.num_layers > 0:
            self.video_encoder = Encoder(hidden_channels, drop_rate, cfg.arch.embedder_model.num_heads, 
                                            cfg.arch.embedder_model.d_ff, cfg.arch.embedder_model.num_layers)
        
        self.embedding_layer = nn.Linear(hidden_channels, self.embedding_size)

    def forward(self, x, video_masks=None):
        batch_size, num_steps, c, h, w = x.shape
        x = x.view(batch_size*num_steps, c, h, w)

        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        x = self.video_emb(x)
        x = x.view(batch_size, num_steps, x.size(1))
        x = self.video_pos_enc(x)
        if self.cfg.arch.embedder_model.num_layers > 0:
            x = self.video_encoder(x, src_mask=video_masks)

        x = x.view(batch_size*num_steps, -1)
        x = self.embedding_layer(x)
        x = x.view(batch_size, num_steps, self.embedding_size)
        return x
    
class MLPHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        projection_hidden_size = cfg.arch.projection_size
        self.embedding_size = cfg.arch.embedder_model.embedding_size
        self.net = nn.Sequential(nn.Linear(self.embedding_size, projection_hidden_size),
                                nn.BatchNorm1d(projection_hidden_size),
                                nn.ReLU(True),
                                nn.Linear(projection_hidden_size, self.embedding_size))
    
    def forward(self, x):
        b, l, c = x.shape
        x = x.view(-1,c)
        x = self.net(x)
        return x.view(b, l, c)

class Classifier(nn.Module):
    """Classifier network.
    """
    def __init__(self, cfg):
        super(Classifier, self).__init__()

        # if cfg.DATASETS[0] == "finegym":
        #     self.num_classes = cfg.EVAL.CLASS_NUM
        # else:
        #     self.num_classes = DATASET_TO_NUM_CLASSES[cfg.DATASETS[0]]
        self.embedding_size = cfg.arch.embedder_model.embedding_size
        drop_rate = cfg.arch.embedder_model.dropout

        self.fc_layers = []
        self.fc_layers.append(nn.Dropout(drop_rate))
        self.fc_layers.append(nn.Linear(self.embedding_size, self.num_classes))
        self.fc_layers = nn.Sequential(*self.fc_layers)

    def forward(self, x):
        # Pass through fully connected layers.
        x = self.fc_layers(x)
        return x