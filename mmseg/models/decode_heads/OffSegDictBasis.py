# -*- coding: utf-8 -*-
"""A shared dictionary of residual directions instead of 150 private bases.

What changes
------------
The residual subspace of class k is currently a private parameter block
`U_k in R^(C x r)`; with K=150, C=256, r=4 that is 153,600 parameters and it
is the overwhelming majority of the whole module (about 0.26M).  Every class
learns its four directions alone, from only the pixels of that class.

Here the directions come from a shared dictionary::

    atoms  D in R^(m x C)        m shared directions in feature space
    coeff  A_k in R^(m x r)      each class picks a rank-r combination
    U_k    = orthonormalise(D^T A_k)

At m=64 that is 16,384 + 38,400 = 54,784 parameters against 153,600 -- about
2.8x fewer -- and, more importantly, every class's directions are estimated
from ALL classes' gradients instead of its own alone.

Why this, and why now
---------------------
* Direct evidence that per-class freedom overfits here: RGE + class-wise
  grouped SE read 46.49, i.e. -1.07 against the shared-excitation version.
  The basis is the largest purely per-class parameter block in the model and
  has never been shared.
* Direct evidence from the measured Stuff generalisation: the paired gain is
  +0.42 at T but only +0.07 at B, and Stuff has 171 classes at half the ADE
  schedule -- exactly the regime where per-class blocks get the least data
  each.  If the bases are undertrained at scale, sharing is the fix.
* It is a subtraction.  Every capacity-adding experiment on this scorer has
  lost points; the only cheap structural change so far (RGE, matrix removed)
  cost just 0.23.

Nothing else moves: projection, per-image second moment, competitive
responsibility, stop-gradient, scale and mix are inherited unchanged, and no
loss is added.  `dict_size` is the single new hyper-parameter.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegACS import ImageAdaptiveAffineClassSubspace, OffSegCCMIACS


class DictionaryAffineClassSubspace(ImageAdaptiveAffineClassSubspace):
    """IACS whose per-class basis is a combination of shared atoms."""

    def __init__(self, *args, dict_size: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        atoms = int(dict_size)
        if atoms < self.rank:
            raise ValueError(
                'dict_size must be at least rank, got %d < %d'
                % (atoms, self.rank))
        self.dict_size = atoms

        # Drop the private [K, C, r] block.  A one-element buffer keeps the
        # name alive because the inherited helpers use `raw_basis.new_*`
        # purely as a dtype/device anchor.
        del self.raw_basis
        self.register_buffer(
            'raw_basis', torch.zeros(1), persistent=False)

        self.dict_atoms = nn.Parameter(torch.empty(atoms, self.embed_dims))
        self.dict_coeff = nn.Parameter(
            torch.empty(self.num_classes, atoms, self.rank))
        # Match the scale the private basis used to start at, so the residual
        # projections keep the same magnitude at initialisation.
        nn.init.normal_(self.dict_atoms, std=1.0 / math.sqrt(self.embed_dims))
        nn.init.normal_(self.dict_coeff, std=1.0 / math.sqrt(atoms))

    def orthonormal_basis(self):
        """Combine the shared atoms per class, then Gram-Schmidt as before."""
        raw = torch.einsum(
            'kmr,mc->kcr', self.dict_coeff, self.dict_atoms)     # [K,C,r]
        vectors = []
        for index in range(self.rank):
            vector = raw[:, :, index]
            for previous in vectors:
                vector = vector - (vector * previous).sum(
                    dim=1, keepdim=True) * previous
            vector = F.normalize(vector, dim=1, eps=self.eps)
            vectors.append(vector)
        return torch.stack(vectors, dim=-1)                      # [K,C,r]

    def atom_usage(self):
        """How many atoms each class actually leans on (participation ratio).

        Near `dict_size` means the classes spread over the whole dictionary;
        near `rank` means each class collapsed onto its own few atoms and the
        sharing is nominal only.
        """
        weight = self.dict_coeff.detach().pow(2).sum(dim=-1)     # [K,m]
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return weight.sum(dim=-1).pow(2) / weight.pow(2).sum(
            dim=-1).clamp_min(self.eps)


@MODELS.register_module()
class OffSegCCMIACSDict(OffSegCCMIACS):
    """Responsibility-IACS over a shared residual-direction dictionary."""

    def __init__(self, in_channels, new_channels, num_classes,
                 acs_rank=4, acs_scale_init=0.05, dict_size=64, **kwargs):
        super().__init__(
            in_channels=in_channels, new_channels=new_channels,
            num_classes=num_classes, acs_rank=acs_rank,
            acs_scale_init=acs_scale_init, **kwargs)
        previous = self.acs
        self.acs = DictionaryAffineClassSubspace(
            num_classes=self.num_classes,
            embed_dims=self.channels,
            rank=previous.rank,
            scale_init=float(acs_scale_init),
            mix_init=float(torch.sigmoid(previous.mix_logit.detach()).item()),
            scatter_eps=previous.scatter_eps,
            detach_statistics=previous.detach_statistics,
            classwise_mix=previous.classwise_mix,
            center_statistics=previous.center_statistics,
            assignment=previous.assignment,
            reliability_shrink=previous.reliability_shrink,
            persistent_spectrum=previous.spectrum_raw is not None,
            spectrum_scale=previous.spectrum_scale,
            learn_competition_strength=previous.competition_raw is not None,
            competition_bound=previous.competition_bound,
            dict_size=int(dict_size))

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_dict_atom_usage'] = self.acs.atom_usage().mean()
        return losses
