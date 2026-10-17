"""Observe existing Route calls without substituting any forward result."""
import numpy as np
from PIL import Image
from .common import crop_boxes


def align_gt(gt, shape, padding, input_shape):
    """Original GT -> resized evaluation input, then explicit ignored padding."""
    h, w = map(int, shape[:2])
    left, right, top, bottom = map(int, padding)
    result = np.array(Image.fromarray(gt.astype('uint8')).resize((w, h), Image.Resampling.NEAREST))
    result = np.pad(result, ((top, bottom), (left, right)), constant_values=255)
    if result.shape != tuple(input_shape):
        raise ValueError(f'GT/input geometry mismatch: {result.shape} != {input_shape}; no guessed heatmaps')
    return result


class RouteCapture:
    def __init__(self, model, dense=False, class_ids=()):
        self.model, self.head = model, model.decode_head
        self.dense, self.class_ids = dense, list(class_ids)
        self.windows = []
        self.handles = []

    def install(self):
        import torch
        self.original_inference = self.model.inference
        self.original_encode = self.model.encode_decode
        self.original_blend = self.head._blend_prototypes
        self.original_correction = self.head._subspace_correction
        self.original_metric = self.head.acs.image_metric

        def inference(inputs, metas):
            if len(metas) != 1 or inputs.shape[0] != 1 or metas[0].get('flip', False):
                raise ValueError('Evidence export requires batch 1, no TTA/flip')
            self.windows = []
            self.geometry = dict(img_shape=tuple(metas[0]['img_shape'][:2]),
                                 padding=tuple(metas[0].get('padding_size', (0, 0, 0, 0))),
                                 input_shape=tuple(inputs.shape[-2:]))
            cfg = self.model.test_cfg
            if cfg['mode'] != 'slide':
                raise ValueError('Use the unchanged ADE slide protocol')
            self.boxes = crop_boxes(*inputs.shape[-2:], cfg['crop_size'], cfg['stride'])
            self.next_window = 0
            result = self.original_inference(inputs, metas)
            if self.next_window != len(self.boxes):
                raise RuntimeError('Observed crop calls do not match the original slide grid')
            return result

        def encode(inputs, metas):
            self.current = dict(box=self.boxes[self.next_window])
            self.next_window += 1
            if self.dense and hasattr(self.model, 'data_preprocessor'):
                pre = self.model.data_preprocessor
                rgb = inputs[0].detach().clone()
                if pre._enable_normalize:
                    rgb = rgb * pre.std + pre.mean
                if not pre.channel_conversion:
                    rgb = rgb[[2,1,0]]
                self.current['rgb_input'] = rgb.permute(1,2,0).round().clamp(0,255).byte().cpu().numpy()
            result = self.original_encode(inputs, metas)
            self.windows.append(self.current)
            return result

        def blend(masks, centres):
            result = self.original_blend(masks, centres)
            with torch.no_grad():
                support = masks.detach().float().softmax(1).sum(-1)
                n0 = torch.nn.functional.softplus(self.head.proto_n0_raw.detach())
                lam = n0 / (support.to(centres.dtype) + n0)
                if self.head.proto_fixed_lambda > 0:
                    lam = torch.full_like(lam, self.head.proto_fixed_lambda)
                lam *= (self.head.proto_seen > 0).view(1, -1)
                self.current.update(support=support[0].cpu().numpy(),
                                    lam=lam[0].float().cpu().numpy())
            return result

        def ccm_hook(module, args, output):
            if not self.dense:
                return
            with torch.no_grad():
                probability = args[2].detach().transpose(1, 2).softmax(-1)
                keep = module._nucleus(probability)
                if keep.dtype != torch.bool:
                    raise ValueError('Expected original nucleus 0.9 boolean candidate mask')
                self.current['candidate_count'] = keep[0].sum(-1).cpu().numpy().astype('uint8')

        def correction(feat, centres, logits, spatial_shape=None):
            result = self.original_correction(feat, centres, logits, spatial_shape=spatial_shape)
            self.current['feature_shape'] = tuple(spatial_shape)
            if self.dense:
                with torch.no_grad():
                    ids = self.class_ids
                    raw = (feat @ centres.transpose(1, 2)).detach()
                    delta = result[0].detach()
                    weights = self.head.acs.assignment_weights(logits.detach())
                    spatial = logits.detach().softmax(dim=1)
                    top = raw.topk(2, dim=-1).indices
                    competitors = torch.stack([torch.where(top[..., 0] == k, top[..., 1], top[..., 0]) for k in ids], -1)
                    self.current.update(
                        class_ids=np.asarray(ids, dtype='int64'),
                        responsibility=weights[0, :, ids].transpose(0, 1).cpu().numpy(),
                        spatial_weight=spatial[0, :, ids].transpose(0, 1).cpu().numpy(),
                        delta=delta[0, :, ids].transpose(0, 1).cpu().numpy(),
                        delta_margin=(delta[..., ids] - delta.gather(-1, competitors))[0].transpose(0, 1).cpu().numpy(),
                        competitor=competitors[0].transpose(0, 1).cpu().numpy().astype('uint8'),
                        before=raw[0].argmax(-1).cpu().numpy().astype('uint8'),
                        after=(raw+delta)[0].argmax(-1).cpu().numpy().astype('uint8'))
            return result

        def image_metric(projection, logits):
            result = self.original_metric(projection, logits)
            if self.dense:
                self.current['metric4x4'] = result[0][0, self.class_ids].detach().cpu().numpy()
                self.current['metric_mix'] = result[1].detach().cpu().numpy()
            return result

        self.model.inference, self.model.encode_decode = inference, encode
        self.head._blend_prototypes, self.head._subspace_correction = blend, correction
        self.head.acs.image_metric = image_metric
        self.handles.append(self.head.ccm.register_forward_hook(ccm_hook))

    def consume(self, gt):
        aligned = align_gt(gt, self.geometry['img_shape'], self.geometry['padding'], self.geometry['input_shape'])
        classes = self.head.num_classes
        for window in self.windows:
            y1, x1, y2, x2 = window['box']
            crop_gt = aligned[y1:y2, x1:x2]
            counts = np.bincount(crop_gt[crop_gt < classes], minlength=classes)
            window['gt_area'] = counts.astype('int64')
            window['valid_pixels'] = int(counts.sum())
            if self.dense:
                h, w = window['feature_shape']
                window['gt_grid'] = np.array(Image.fromarray(crop_gt).resize((w, h), Image.Resampling.NEAREST))
        return self.windows

    def close(self):
        self.model.inference, self.model.encode_decode = self.original_inference, self.original_encode
        self.head._blend_prototypes, self.head._subspace_correction = self.original_blend, self.original_correction
        self.head.acs.image_metric = self.original_metric
        for handle in self.handles:
            handle.remove()
        self.handles = []
