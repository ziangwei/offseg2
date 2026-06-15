import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .freqfusion import FreqFusion
from mmseg.models.decode_heads import Offset_Learning
from .MaskTransformer3 import SpatialAttributeDecoder
import numpy as np
import cv2
import math

# def cv_squared(x):
#     """The squared coefficient of variation of a sample."""
#     eps = 1e-10
#     if x.shape[0] == 1:
#         return torch.tensor([0], device=x.device, dtype=x.dtype)
#     return x.float().var() / (x.float().mean() ** 2 + eps)


class AttentionGatedCorrectionFusion(nn.Module):
    def __init__(self, num_classes, hidden=32):
        super().__init__()
        self.num_classes = num_classes

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True)
        )

        mid_channels = max(num_classes // 8, 4)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_classes * 2, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True)
        )
        self.channel_floor_logit = nn.Parameter(torch.tensor(0.0))

        nn.init.constant_(self.spatial_attn[-1].bias, -2.0)
        nn.init.constant_(self.channel_attn[-1].bias, -2.0)
        self.register_buffer(
            "max_entropy",
            torch.tensor(math.log(num_classes), dtype=torch.float32),
            persistent=False
        )

    def forward(self, base_head_logits, refinement_head_logits):
        p_base = F.softmax(base_head_logits, dim=1)
        p_refinement = F.softmax(refinement_head_logits, dim=1)
        max_ent = self.max_entropy.to(device=base_head_logits.device, dtype=base_head_logits.dtype)

        entropy_base = -(p_base * torch.log(p_base.clamp_min(1e-6))).sum(dim=1, keepdim=True)
        entropy_refinement = -(p_refinement * torch.log(p_refinement.clamp_min(1e-6))).sum(dim=1, keepdim=True)
        entropy_base = (entropy_base / (max_ent + 1e-6)).clamp(0.0, 1.0)
        entropy_refinement = (entropy_refinement / (max_ent + 1e-6)).clamp(0.0, 1.0)

        disagree = 0.5 * torch.sum(torch.abs(p_base - p_refinement), dim=1, keepdim=True)

        spatial_alpha = torch.sigmoid(
            self.spatial_attn(torch.cat([entropy_base, entropy_refinement, disagree], dim=1))
        )   # [B,1,H,W]

        channel_alpha = torch.sigmoid(
            self.channel_attn(torch.cat([base_head_logits, refinement_head_logits], dim=1))
        )   # [B,C,1,1]
        floor = torch.sigmoid(self.channel_floor_logit)
        alpha = spatial_alpha * (floor + (1.0 - floor) * channel_alpha)

        #alpha = spatial_alpha * channel_alpha
        fuse_logits = base_head_logits + alpha * (refinement_head_logits - base_head_logits)
        return fuse_logits


class PrototypeGuidedAttributeCalibration(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        cls_attributes,
        residual_scale=0.2,
        topk_div=64,
        sparse_proto=True,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.cls_attributes = cls_attributes
        self.residual_scale = residual_scale
        self.topk_div = topk_div
        self.sparse_proto = sparse_proto

        self.proto_proj = nn.Linear(dim, dim)

        hidden = dim // 4
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )

        nn.init.uniform_(self.gate_mlp[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)

        self.norm = nn.LayerNorm(dim)

        self.register_buffer(
            "max_entropy",
            torch.tensor(math.log(num_classes), dtype=torch.float32),
            persistent=False
        )

    def forward(self, attr_tokens, refinement_feats, base_head_logits):
        B, Nc, A, D = attr_tokens.shape

        
        base_head_logits = base_head_logits.detach()
        p_base = F.softmax(base_head_logits, dim=1)
            

        logp = torch.log(p_base.clamp_min(1e-6))
        entropy = -(p_base * logp).sum(dim=1, keepdim=True)
        entropy = entropy / (self.max_entropy.to(dtype=entropy.dtype) + 1e-6)
        entropy = entropy.clamp(0.0, 1.0)

        confidence = 1.0 - entropy

        # [B, Nc, H, W]
        class_mask = p_base * confidence

        # [B, Nc, HW]
        mask_flat = class_mask.flatten(2)

        hw = mask_flat.shape[-1]
        k = max(1, hw // self.topk_div)


        topk_vals, topk_idx = torch.topk(mask_flat, k=k, dim=-1)

        sparse_mask = torch.zeros_like(mask_flat)
        sparse_mask.scatter_(-1, topk_idx, topk_vals)

        proto_weight = sparse_mask

        proto_weight_sum = proto_weight.sum(dim=-1, keepdim=True)
        proto_weight_norm = proto_weight / (proto_weight_sum + 1e-6)

        feat_flat = refinement_feats.flatten(2).transpose(1, 2)   # [B, HW, D]
        class_proto = torch.bmm(proto_weight_norm, feat_flat)   # [B, Nc, D]

        topk_vals_presence = torch.topk(mask_flat, k=k, dim=-1).values
        presence = topk_vals_presence.mean(dim=-1, keepdim=True)   # [B, Nc, 1]

        proto_base = self.proto_proj(class_proto)            # [B, Nc, D]
        proto_proj = proto_base.unsqueeze(2).expand(-1, -1, A, -1)

        gate_input = torch.cat([
            attr_tokens,
            proto_proj,
            torch.abs(attr_tokens - proto_proj)
        ], dim=-1)

        gate = torch.sigmoid(self.gate_mlp(gate_input))      # [B, Nc, A, 1]
        presence_attr = presence.unsqueeze(2)                # [B, Nc, 1, 1]

        calibrated_attr_tokens = self.norm(
            attr_tokens
            + self.residual_scale
            * presence_attr
            * gate
            * (proto_proj - attr_tokens)
        )

        return calibrated_attr_tokens, class_proto    

class PrototypeAttributeRefinementHead(nn.Module):
    def __init__(self,
                 in_channels, 
                 num_classes, 
                 cls_attributes, 
                 mask_dim=256, 
                 args=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        self.mask_dim = mask_dim
        self.args = args or {}
        self.refinement_feat_proj = ConvModule(
            in_channels,
            mask_dim,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.spatial_attribute_decoder = SpatialAttributeDecoder(
            in_channels=mask_dim, 
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            mask_dim=mask_dim
        )

        route_hidden = mask_dim // 4

        self.route_mlp = nn.Sequential(
            nn.Linear(mask_dim, route_hidden),
            nn.LayerNorm(route_hidden),
            nn.GELU(),
            nn.Linear(route_hidden, cls_attributes)
        )

        nn.init.uniform_(self.route_mlp[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.route_mlp[-1].bias)
        self.route_class_bias = nn.Embedding(num_classes, cls_attributes)

        nn.init.zeros_(self.route_class_bias.weight)

        # self.fc1 = nn.Linear(in_features=mask_dim, out_features=mask_dim)
        # self.fc2 = nn.Linear(in_features=mask_dim, out_features=mask_dim)
        # self.LeakyReLU = nn.LeakyReLU(0.2)

        self.feat_norm = nn.LayerNorm(mask_dim)
    
        self.proto_refiner = PrototypeGuidedAttributeCalibration(
            dim=mask_dim,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            residual_scale=self.args['proto_residual_scale'],
            topk_div=self.args['proto_topk_div'],  
        )

    
    def forward(self, feat_aligned, base_head_logits):
        refinement_feats = self.refinement_feat_proj(feat_aligned) #(B, mask_dim, H, W)

        attr_tokens = self.spatial_attribute_decoder(refinement_feats=refinement_feats, base_head_logits=base_head_logits) #(B, Nc, A, D)

        calibrated_attr_tokens, class_proto = self.proto_refiner(
            attr_tokens=attr_tokens,
            refinement_feats=refinement_feats,
            base_head_logits=base_head_logits
        )

        route_input = class_proto.detach()                    # [B, Nc, D]
        dynamic_route = self.route_mlp(route_input)          # [B, Nc, A]

        class_bias = self.route_class_bias.weight.unsqueeze(0)   # [1, Nc, A]
        route_value = dynamic_route + class_bias

        route_prob = F.softmax(route_value, dim=-1)     # soft routing

        if self.args['use_class_prototypes']:
            class_feats = torch.einsum('bcad,bca->bcd', calibrated_attr_tokens, route_prob) # [B, Nc, D]
        else:
            class_feats = torch.einsum('bcad,bca->bcd', attr_tokens, route_prob) # [B, Nc, D]
        #class_feats = self.fc2(self.LeakyReLU(self.fc1(class_feats)))   # [B, Nc, D]        

        seg_feats = refinement_feats.permute(0, 2, 3, 1)            # [B, H, W, D]

        #seg_feats = self.feat_norm(seg_feats)
        seg_feats = F.normalize(seg_feats, p=2, dim=-1, eps=1e-6)
        class_feats = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)

        class_pixel_sim = torch.einsum("bhwd,bcd->bchw", seg_feats, class_feats)

        refinement_head_logits = class_pixel_sim / self.args['tau']

        return refinement_head_logits, calibrated_attr_tokens


@MODELS.register_module()
class PARSeg3(BaseDecodeHead):
    def __init__(self, 
                 in_channels,
                 new_channels,
                 num_classes,
                 cls_attributes,
                 args=None,
                 **kwargs):
        super().__init__(in_channels=in_channels, 
                         num_classes=num_classes, 
                         input_transform='multiple_select', 
                         **kwargs)
        self.new_channels = new_channels

        self.args = args

        self.pre = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.pre.append(
                ConvModule(self.in_channels[i],
                           self.new_channels[i],
                           1,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg)
            )
        
        self.freqfusions = nn.ModuleList()
        in_channels = new_channels[::-1]
        pre_c = in_channels[0]
        for c in in_channels[1:]:
            freqfusion = FreqFusion(
                hr_channels=c, lr_channels=pre_c, 
                compressed_channels=(pre_c + c) // 4,
            )                
            self.freqfusions.append(freqfusion)
            pre_c += c

        self.align = ConvModule(
            sum(self.new_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.offset_learning = Offset_Learning(self.num_classes, self.channels)

        self.prototype_attribute_refinement = PrototypeAttributeRefinementHead(
            in_channels=self.channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        self.fusion = AttentionGatedCorrectionFusion(num_classes=num_classes)
        self.fuse_catconv = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, bias=True)

    def forward(self, inputs, return_vis=False):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        new_inputs = []
        for i in range(len(inputs)):
            new_inputs.append(self.pre[i](inputs[i]))

        lowres_feat = new_inputs[-1] # small map
        for idx, (hires_feat, freqfusion) in enumerate(zip(new_inputs[:-1][::-1], self.freqfusions)):

            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat([hires_feat.reshape(b * 4, -1, h, w), 
                                    lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)
        
        aligned_inputs = lowres_feat # High Res Fused Feature
        
        feat_aligned = self.align(aligned_inputs)
        base_head_logits = self.offset_learning(feat_aligned) # Logits (B, 150, H, W)

        refinement_head_logits, calibrated_attr_tokens = self.prototype_attribute_refinement(feat_aligned, base_head_logits)
        fusion_mode = self.args.get('fusion_mode', 'AGC')
        if fusion_mode == 'AGCF':
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == 'avg':
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == 'catconv':
            final_logits = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))

        returndict = {}
        returndict['base_head_logits'] = base_head_logits
        returndict['calibrated_attr_tokens'] = calibrated_attr_tokens
        returndict['refinement_head_logits'] = refinement_head_logits
        #returndict['route_prob'] = route_prob
        returndict['final_logits'] = final_logits

        return returndict
    
    def predict(self, inputs, batch_img_metas, test_cfg, **kwargs):
        returndict = self.forward(inputs)
        seg_logits = returndict['final_logits']
        return self.predict_by_feat(seg_logits, batch_img_metas)

    def _dynamic_class_weight(self, seg_label):
        device = seg_label.device
        valid = seg_label != self.ignore_index

        if valid.sum() == 0:
            return torch.ones(self.num_classes, device=device)

        label_valid = seg_label[valid].long()

        counts = torch.bincount(
            label_valid,
            minlength=self.num_classes
        ).float().to(device)

        present = counts > 0
        freq = counts / counts.sum().clamp_min(1.0)

        class_w = torch.ones(self.num_classes, device=device)

        if present.sum() > 0:
            mean_freq = freq[present].mean().clamp_min(1e-6)
            class_w[present] = torch.sqrt(
                mean_freq / freq[present].clamp_min(1e-6)
            )

        class_w = class_w.clamp(0.5, 3.0)

        return class_w
    

    def _base_error_focused_ce(
        self,
        logits,
        seg_label,
        base_head_logits,
        err_weight=1.0,
        unc_weight=0.5,
        use_class_balance=True,
    ):
        valid = seg_label != self.ignore_index

        with torch.no_grad():
            p_base = F.softmax(base_head_logits.detach(), dim=1)
            base_conf, base_pred = p_base.max(dim=1)  # [B, H, W]

            base_wrong = ((base_pred != seg_label) & valid).float()

            base_uncertain = (1.0 - base_conf).clamp(0.0, 1.0)

            pixel_w = 1.0 + err_weight * base_wrong + unc_weight * base_uncertain

            if use_class_balance:
                class_w = self._dynamic_class_weight(seg_label)

                safe_label = seg_label.clone()
                safe_label[~valid] = 0

                class_w_map = class_w[safe_label.long()]  # [B, H, W]
                pixel_w = pixel_w * class_w_map

            pixel_w = pixel_w * valid.float()

            pixel_w = pixel_w.clamp(0.0, 4.0)

        ce = F.cross_entropy(
            logits,
            seg_label.long(),
            ignore_index=self.ignore_index,
            reduction='none'
        )  # [B, H, W]

        loss = (ce * pixel_w).sum() / pixel_w.sum().clamp_min(1.0)

        return loss
    
    def loss_by_feat(self, seg_logits, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples)
        
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
            
        target_size = seg_label.shape[-2:]

        losses = dict()
        
        base_pred = seg_logits['base_head_logits']  
        refinement_pred = seg_logits['refinement_head_logits']   
        final_pred = seg_logits['final_logits'] 
        calibrated_attr_tokens = seg_logits['calibrated_attr_tokens'] 

        if base_pred is not None:
            base_pred_resized = resize(
                input=base_pred,
                size=target_size,
                mode='bilinear',
                align_corners=self.align_corners)
            
            losses['loss_base'] = self.loss_decode(
                base_pred_resized, 
                seg_label, 
                ignore_index=self.ignore_index) * self.args['basew']

        if refinement_pred is not None:
            refinement_pred_resized = resize(
                input=refinement_pred,
                size=target_size,
                mode='bilinear',
                align_corners=self.align_corners)
            
            losses['loss_refinement'] = self.loss_decode(
                refinement_pred_resized, 
                seg_label, 
                ignore_index=self.ignore_index) * self.args['refinementw']
        
        if final_pred is not None:
            fuse_pred_resized = resize(
                input=final_pred,
                size=target_size,
                mode='bilinear',
                align_corners=self.align_corners)
                
            losses['loss_fusion'] = self.loss_decode(
                fuse_pred_resized, 
                seg_label, 
                ignore_index=self.ignore_index) * self.args['fusionw']
            
        refinement_focusw = self.args.get('refinement_focusw', 0.25)

        if refinement_focusw > 0:
            losses['loss_refinement_focus'] = self._base_error_focused_ce(
                logits=refinement_pred_resized,
                seg_label=seg_label,
                base_head_logits=base_pred_resized,
                err_weight=self.args.get('focus_err_weight', 1.0),
                unc_weight=self.args.get('focus_unc_weight', 0.5),
                use_class_balance=self.args.get('focus_class_balance', True),
            ) * refinement_focusw

        B, Nc, A, D = calibrated_attr_tokens.shape
        device = calibrated_attr_tokens.device

        with torch.no_grad():
            gt_present = torch.zeros(B, Nc, device=device, dtype=torch.bool)
            for b in range(B):
                present_cls = torch.unique(seg_label[b])
                present_cls = present_cls[present_cls != self.ignore_index]
                if present_cls.numel() > 0:
                    gt_present[b, present_cls.long()] = True

        present_idx = gt_present.view(-1).nonzero(as_tuple=False).squeeze(1)

        if present_idx.numel() > 0:
            att_flat = calibrated_attr_tokens.reshape(B * Nc, A, D)
            att_valid = att_flat.index_select(0, present_idx)      # [N_valid, A, D]
            att_valid = F.normalize(att_valid, p=2, dim=-1)

            gram_valid = torch.bmm(att_valid, att_valid.transpose(1, 2))   # [N_valid, A, A]

            eye = torch.eye(A, device=device, dtype=gram_valid.dtype).unsqueeze(0)
            off_diag_mask = 1.0 - eye

            loss_intra_div = ((gram_valid * off_diag_mask) ** 2).mean()
        else:
            loss_intra_div = calibrated_attr_tokens.sum() * 0.0

        losses['loss_intra_div'] = loss_intra_div * self.args['intra_div']

        # route_prob = seg_logits.get('route_prob')
        # if route_prob is not None and self.args.get('loadbal', 0) > 0:
        #     present_weight = gt_present.unsqueeze(-1).to(route_prob.dtype)   # [B,Nc,1]
        #     usage_per_expert = (route_prob * present_weight).sum(dim=(0, 1)) # [A]
        #     loss_loadbal = cv_squared(usage_per_expert)
        #     losses['loss_loadbal'] = loss_loadbal * self.args['loadbal']
        return losses
