# -*- coding: utf-8 -*-
"""Encode the per-class attribute DESCRIPTIONS into a frozen CLIP asset.

Companion to tools/gen_text_anchors.py, but deliberately different in two
ways (both are lessons from the name-anchor failure, see experiment log):
  * encodes DESCRIPTIONS ONLY -- the class name is NOT inserted into the
    prompt, so the embedding carries attribute content, not name geometry
    (the name cone is what crashed LTA/PTA);
  * embeddings are kept as a PER-CLASS SET [150, K, 512] and NEVER averaged
    (averaging K descriptions would re-create a cone).

Run OFFLINE once (server is fine; ~600MB CLIP download into --cache-dir):
    HF_HOME=/path/to/cache python tools/gen_text_descriptions.py
Output: assets/text_anchors/ade20k_clip_vitb32_desc6.pt  (~1.9 MB)
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_text_anchors import ADE_CLASSES               # exact official order
from ade20k_class_descriptions import DESCRIPTIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out", default=os.path.join(
        "assets", "text_anchors", "ade20k_clip_vitb32_desc6.pt"))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    ks = {len(v) for v in DESCRIPTIONS.values()}
    assert len(ks) == 1, f"uneven description counts: {ks}"
    K = ks.pop()
    assert set(DESCRIPTIONS) == set(ADE_CLASSES), "class-name mismatch"

    from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    tok = CLIPTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    enc = CLIPTextModelWithProjection.from_pretrained(
        args.model, cache_dir=args.cache_dir).to(args.device).eval()

    prompts = [desc for name in ADE_CLASSES for desc in DESCRIPTIONS[name]]
    feats = []
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch):
            batch = tok(prompts[i:i + args.batch], padding=True,
                        return_tensors="pt").to(args.device)
            out = enc(**batch).text_embeds
            feats.append(torch.nn.functional.normalize(out, dim=-1).cpu())
    embs = torch.cat(feats).reshape(len(ADE_CLASSES), K, -1)   # [150, K, 512]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "embeddings": embs,                    # [150, K, 512], per-desc L2
        "class_names": ADE_CLASSES,
        "descriptions": {n: DESCRIPTIONS[n] for n in ADE_CLASSES},
        "model": args.model,
        "note": "attribute-description embeddings, never averaged; "
                "generated offline by tools/gen_text_descriptions.py",
    }, args.out)
    print(f"saved {tuple(embs.shape)} -> {args.out}")

    # geometry eyeball: description sets should be far less cone-collapsed
    # than name anchors were (that is the whole point)
    flat = torch.nn.functional.normalize(embs.mean(dim=1), dim=-1)
    sim = flat @ flat.t()
    off = sim[~torch.eye(len(ADE_CLASSES), dtype=torch.bool)]
    print(f"class-mean pairwise cos: mean={off.mean():.3f} "
          f"p95={off.quantile(0.95):.3f} "
          "(name anchors were mean 0.782 / confusion pairs 0.841)")


if __name__ == "__main__":
    main()
