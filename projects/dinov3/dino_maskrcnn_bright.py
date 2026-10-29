#!/usr/bin/env python3
"""Train DINOv3 + FPN + Mask R-CNN on BRIGHT instance annotations."""
from __future__ import annotations

import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from src.dataset.data import BRIGHTDataset, get_transforms
from src.engine import evaluate, format_metrics_report, train_one_epoch
from semantic_dinov3.model import DINOV3Backbone


class DINOFPNBackbone(nn.Module):
    """Expose DINO intermediate token maps through a torchvision FPN."""
    def __init__(self, model_name, in_chans=4, pretrained=True, fpn_channels=256):
        super().__init__()
        self.dino = DINOV3Backbone(model_name, in_chans=in_chans, pretrained=pretrained)
        self.embed_dim = self.dino.embed_dim
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[self.embed_dim] * len(self.dino.feature_blocks),
            out_channels=fpn_channels,
        )
        self.out_channels = fpn_channels

    def forward(self, x):
        feats = self.dino(x)
        return self.fpn(OrderedDict((str(i), feat) for i, feat in enumerate(feats)))


def build_model(model_name, in_chans=4, pretrained=True, fpn_channels=256):
    backbone = DINOFPNBackbone(model_name, in_chans, pretrained, fpn_channels)
    # DINO patch-16 maps are stride 16; extra pyramid levels handle small objects.
    anchor_sizes = ((16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    model = MaskRCNN(
        backbone, num_classes=4, rpn_anchor_generator=anchor_generator,
        box_detections_per_img=1000, rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_test=1000,
        image_mean=[0.485, 0.456, 0.406, 0.2443],
        image_std=[0.229, 0.224, 0.225, 0.1718],
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, 4)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--train-ann", required=True)
    p.add_argument("--val-ann", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="vit_base_patch16_dinov3")
    p.add_argument("--channels", type=int, choices=(3, 4), default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--resume", default="")
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    image_dir = os.path.join(args.root, "post-event")
    pre_dir = os.path.join(args.root, "pre-event")
    train_ds = BRIGHTDataset(args.train_ann, image_dir, pre_dir, get_transforms(True))
    val_ds = BRIGHTDataset(args.val_ann, image_dir, pre_dir, get_transforms(False))
    collate = lambda batch: tuple(zip(*batch))
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, 1, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate, pin_memory=True)
    model = build_model(args.model_name, args.channels, not args.no_pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    start = 0
    best = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"]); start = ckpt["epoch"] + 1
        best = ckpt.get("best_ap", -1.0)
    log_path = os.path.join(args.output_dir, "train.log")
    for epoch in range(start, args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20,
                        scaler=scaler, warmup_iters=500 if epoch == 0 else 0,
                        warmup_factor=0.01, log_file=log_path)
        metrics = evaluate(model, val_loader, device, output_dir=os.path.join(args.output_dir, "eval"),
                           epoch=epoch, log_file=log_path)
        print(f"Epoch {epoch}: " + " | ".join(format_metrics_report(metrics)), flush=True)
        scheduler.step()
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "metrics": metrics, "best_ap": best}
        if scaler is not None: ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, os.path.join(args.output_dir, "latest.pth"))
        ap = metrics.get("segm_AP", 0.0)
        if ap > best:
            best = ap; ckpt["best_ap"] = best
            torch.save(ckpt, os.path.join(args.output_dir, "best_model.pth"))
            print(f"[save] best segm_AP={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
