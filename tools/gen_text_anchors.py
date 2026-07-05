# -*- coding: utf-8 -*-
"""Generate frozen CLIP text anchors for the ADE20K-150 vocabulary.

This is the ONLY place any text model ever runs. It executes OFFLINE, ONCE,
and writes a small (~300 KB) .pt asset containing one L2-normalized embedding
per class. Training loads that constant tensor; inference never touches text
in any form. No text encoder is needed on the training server.

Asset management:
  * default output   : assets/text_anchors/ade20k_clip_vitb32.pt (repo root)
  * HF download cache: --cache-dir (default: $HF_HOME or ~/.cache/huggingface)
    On a shared server, point it somewhere with quota, e.g.
        HF_HOME=/data/$USER/hf python tools/gen_text_anchors.py
  * the asset stores class names, templates and the model id, so it is
    self-documenting and reproducible.
  * NOTE: the repo .gitignore excludes *.pt, so commit the asset with
        git add -f assets/text_anchors/ade20k_clip_vitb32.pt

Usage:
    python tools/gen_text_anchors.py                       # all defaults
    python tools/gen_text_anchors.py --device cuda:0       # faster
"""
import argparse
import os

import torch

# Exact mmseg ADE20K-150 names (order matters; some contain legacy trailing
# spaces -- kept verbatim for index alignment, stripped only inside prompts).
ADE_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel",
    "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy",
    "washer", "plaything", "swimming pool", "stool", "barrel", "basket",
    "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle",
    "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce",
    "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen",
    "plate", "monitor", "bulletin board", "shower", "radiator", "glass",
    "clock", "flag",
]

# Scene-oriented prompt ensemble (recorded into the asset for reproducibility).
TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a {} in a room.",
    "a photo of a {} in a scene.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a photo of many {}.",
    "there is a {} in the scene.",
    "a blurry photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
    "a photo of the {} texture.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out", default=os.path.join(
        "assets", "text_anchors", "ade20k_clip_vitb32.pt"))
    ap.add_argument("--cache-dir", default=None,
                    help="HF download cache (default: $HF_HOME or ~/.cache)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    from transformers import CLIPTokenizer, CLIPTextModelWithProjection

    tok = CLIPTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    enc = CLIPTextModelWithProjection.from_pretrained(
        args.model, cache_dir=args.cache_dir).to(args.device).eval()

    prompts, owner = [], []
    for ci, name in enumerate(ADE_CLASSES):
        clean = name.strip().replace("_", " ")
        for t in TEMPLATES:
            prompts.append(t.format(clean))
            owner.append(ci)
    owner = torch.tensor(owner)

    feats = []
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch):
            batch = tok(prompts[i:i + args.batch], padding=True,
                        return_tensors="pt").to(args.device)
            out = enc(**batch).text_embeds                    # [b, 512]
            feats.append(torch.nn.functional.normalize(out, dim=-1).cpu())
    feats = torch.cat(feats)                                  # [150*T, 512]

    embs = torch.zeros(len(ADE_CLASSES), feats.shape[-1])
    for ci in range(len(ADE_CLASSES)):
        embs[ci] = feats[owner == ci].mean(dim=0)
    embs = torch.nn.functional.normalize(embs, dim=-1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "embeddings": embs,                                   # [150, 512], L2-normalized
        "class_names": ADE_CLASSES,
        "templates": TEMPLATES,
        "model": args.model,
        "note": "frozen CLIP text anchors; generated offline by "
                "tools/gen_text_anchors.py; no text model at train/inference",
    }, args.out)
    print(f"saved {tuple(embs.shape)} -> {args.out}")

    # quick geometry eyeball: nearest language neighbours of a few classes
    sim = embs @ embs.t()
    for probe in ["wall", "ceiling", "door", "sidewalk", "armchair"]:
        ci = ADE_CLASSES.index(probe) if probe in ADE_CLASSES else None
        if ci is None:
            continue
        top = sim[ci].topk(4).indices.tolist()[1:]
        print(f"  {probe:10s} ~ " + ", ".join(
            f"{ADE_CLASSES[j].strip()}({sim[ci, j]:.2f})" for j in top))


if __name__ == "__main__":
    main()
