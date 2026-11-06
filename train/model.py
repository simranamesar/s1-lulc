"""U-Net for Sentinel-1 LULC segmentation: CNN encoder (ResNet) + decoder + 1x1 output layer.

Shared by train.py and the prototype, so both build exactly the same network.

Variants (all CNN, no transformer):
    encoder_weights  "imagenet" = transfer learning by full fine-tuning (lecture 08)
                     "none"     = trained from scratch
    freeze_encoder   True = encoder used as a fixed feature extractor (lecture 08), only the decoder learns
    upsample         "interp"     = nearest-neighbour upsampling x2, then convolution (default)
                     "transposed" = learned transposed convolution x2 ("inverse of convolution",
                                    ZFNet deconvolution in lecture 07, Conv2DTranspose in lab 12)

Shapes for ResNet34 and a 256 x 256 input (see the model diagram in the Project Log):
    encoder  3@256 -> 64@128 -> 64@64 -> 128@32 -> 256@16 -> 512@8
    decoder  512@8 -> 256@16 -> 128@32 -> 64@64 -> 32@128 -> 16@256 -> 1x1 conv -> 6@256
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 6   # see data/classes.py
DECODER_CHANNELS = (256, 128, 64, 32, 16)


def conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    """Upsample x2, concatenate the encoder skip of the same size, then two 3x3 conv + BN + ReLU."""

    def __init__(self, cin: int, cskip: int, cout: int, upsample: str):
        super().__init__()
        self.upsample = upsample
        if upsample == "transposed":
            self.up = nn.ConvTranspose2d(cin, cin, kernel_size=2, stride=2)
        elif upsample != "interp":
            raise ValueError(f"unknown upsample mode {upsample!r}")
        self.conv = nn.Sequential(conv_bn_relu(cin + cskip, cout), conv_bn_relu(cout, cout))

    def forward(self, x, skip=None):
        x = self.up(x) if self.upsample == "transposed" else F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegUNet(nn.Module):
    def __init__(self, encoder: str = "resnet34", encoder_weights: str | None = "imagenet",
                 in_channels: int = 3, classes: int = N_CLASSES, upsample: str = "interp",
                 freeze_encoder: bool = False):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.encoder = smp.encoders.get_encoder(encoder, in_channels=in_channels, depth=5, weights=encoder_weights)
        enc = list(self.encoder.out_channels)            # e.g. [3, 64, 64, 128, 256, 512]
        skips = enc[1:-1][::-1] + [0]                    # [256, 128, 64, 64, 0]: no skip at full size
        ins = [enc[-1]] + list(DECODER_CHANNELS[:-1])    # [512, 256, 128, 64, 32]
        self.decoder = nn.ModuleList(DecoderBlock(i, s, o, upsample)
                                     for i, s, o in zip(ins, skips, DECODER_CHANNELS))
        self.head = nn.Conv2d(DECODER_CHANNELS[-1], classes, kernel_size=1)   # 1x1 conv output (lecture 06)

        self.frozen = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()   # fixed feature extractor: keep its batchnorm statistics too
        return self

    def forward(self, x):
        feats = self.encoder(x)              # [input, f1 (1/2), f2 (1/4), f3 (1/8), f4 (1/16), f5 (1/32)]
        x = feats[-1]
        skips = feats[1:-1][::-1] + [None]
        for block, skip in zip(self.decoder, skips):
            x = block(x, skip)
        return self.head(x)


def build_model(encoder: str = "resnet34", encoder_weights: str | None = "imagenet",
                upsample: str = "interp", freeze_encoder: bool = False, classes: int = N_CLASSES) -> SegUNet:
    w = None if encoder_weights in (None, "none", "None") else encoder_weights
    return SegUNet(encoder, w, classes=classes, upsample=upsample, freeze_encoder=freeze_encoder)


def build_from_config(cfg: dict) -> SegUNet:
    """Rebuild the network from a run's config.json (used by the prototype)."""
    return build_model(cfg.get("encoder", "resnet34"), "none",  # weights come from best.pt
                       cfg.get("upsample", "interp"), False, classes=cfg.get("n_classes", 9))
