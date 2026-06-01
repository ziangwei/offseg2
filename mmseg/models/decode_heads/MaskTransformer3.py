# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import math
from .position_encoding import PositionEmbeddingSine


class LinearCrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, spatial_weight=None):
        q = query.transpose(0, 1) # (B, Q_len, D)
        k = key.transpose(0, 1)   # (B, HW_len, D)
        v = value.transpose(0, 1) # (B, HW_len, D)

        B, Q_len, D = q.shape
        HW_len = k.shape[1]

        q = self.q_proj(q).view(B, Q_len, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(k).view(B, HW_len, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(v).view(B, HW_len, self.nhead, self.d_head).transpose(1, 2)

        q = F.softmax(q, dim=-1) 
        k = F.softmax(k, dim=-2) 

        if spatial_weight is not None:
            v = v * spatial_weight

        context = torch.matmul(k.transpose(-1, -2), v)

        out = torch.matmul(q, context) 

        out = out.transpose(1, 2).reshape(B, Q_len, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out.transpose(0, 1)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = LinearCrossAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     spatial_weight = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, spatial_weight=spatial_weight)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = F.relu

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

class SpatialValueWeighting(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        )

        self.prior_logits = nn.Parameter(torch.tensor([0.6, 0.25, 0.15], dtype=torch.float32))

    
    def forward(self, refinement_feats, base_head_logits):

        base_logits_detached = base_head_logits.detach().float()
        log_probs = F.log_softmax(base_logits_detached, dim=1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)

        max_entropy = math.log(base_logits_detached.shape[1])
        entropy_norm = (entropy / (max_entropy + 1e-6)).clamp(0.0, 1.0)

        p_base = F.softmax(base_logits_detached, dim=1)

        # boundary prior from x
        boundary_feat = torch.sigmoid(self.boundary_conv(refinement_feats))
        ks = 3
        pad = ks // 2
        dilate = F.max_pool2d(p_base, kernel_size=ks, stride=1, padding=pad)
        erode = -F.max_pool2d(-p_base, kernel_size=ks, stride=1, padding=pad)
        boundary_prob = (dilate - erode).amax(dim=1, keepdim=True).clamp(0.0, 1.0)

        boundary = 0.5 * boundary_feat + 0.5 * boundary_prob
        #boundary = torch.sigmoid(boundary)

        

        # 方案：top2 margin disagreement
        top2_prob, _ = torch.topk(p_base, k=2, dim=1)         # [B,2,H,W]
        margin = top2_prob[:, 0:1] - top2_prob[:, 1:2]       # [B,1,H,W]
        ambiguity = 1.0 - margin
        ambiguity = ambiguity.clamp(0.0, 1.0)

        coeff = F.softmax(self.prior_logits, dim=0)   
        prior = coeff[0] * entropy_norm + coeff[1] * boundary + coeff[2] * ambiguity

        spatial_weight = 0.1 + 0.9 * prior
        return spatial_weight.to(dtype=refinement_feats.dtype, device=refinement_feats.device)

        

class SpatialAttributeDecoder(nn.Module):
    def __init__(
            self,
            in_channels,
            mask_dim,
            num_classes,
            cls_attributes,
            hidden_dim=256,
            nheads=8,
            dim_feedforward=2048,
    ):
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.num_heads = nheads
        self.hidden_dim = hidden_dim
        self.transformer_cross_attention_layer = CrossAttentionLayer(
                                                        d_model=hidden_dim,
                                                        nhead=nheads,
                                                        dropout=0.0,
                                                    )                
        self.transformer_ffn_layer = FFNLayer(
                                        d_model=hidden_dim,
                                        dim_feedforward=dim_feedforward,
                                        dropout=0.0
                                    )

        self.num_classes = num_classes
        self.cls_attributes = cls_attributes
        self.mask_dim = mask_dim

        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        self.FC_input = nn.Linear(hidden_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.attr_tokens = nn.Linear(hidden_dim, mask_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.query_feat = nn.Embedding(num_classes *cls_attributes, hidden_dim)
        self.query_embed = nn.Embedding(num_classes *cls_attributes, hidden_dim)

        self.spatial_value_weighting = SpatialValueWeighting(in_channels)

    def forward(self, refinement_feats, base_head_logits):
        bs, _, h, w = refinement_feats.shape
        pos = self.pe_layer(refinement_feats, None).flatten(2).permute(2, 0, 1)
        src = self.input_proj(refinement_feats).flatten(2).permute(2, 0, 1)

        q_content = self.query_feat.weight.unsqueeze(1).expand(-1, bs, -1)
        q_pos = self.query_embed.weight.unsqueeze(1).expand(-1, bs, -1)

        spatial_weight = self.spatial_value_weighting(refinement_feats, base_head_logits)
        spatial_weight = spatial_weight.flatten(2).unsqueeze(-1)
        output = self.transformer_cross_attention_layer(
            q_content, src,
            spatial_weight=spatial_weight,
            pos=pos, query_pos=q_pos
        )

        output = self.transformer_ffn_layer(output)
        
        output = output.transpose(0, 1).reshape(bs, self.num_classes, self.cls_attributes, -1)

        h_ = self.LeakyReLU(self.FC_input(output))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        attr_tokens = self.attr_tokens(h_)

        return attr_tokens