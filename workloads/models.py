"""Workload models.

* ``resnet18`` / ``resnet50``: torchvision ResNets with the usual CIFAR stem
  (3x3 stride-1 conv1, no max-pool) for 32x32 inputs.
* ``transformer_small``: decoder-only LM (pre-norm, causal). Alternative for
  when ResNet-18 does not produce a measurable communication regime: more
  gradient bytes per unit of compute.
* ``cnn_tiny``: a few-thousand-parameter CNN for CPU tests only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _cifar_resnet(fn, num_classes: int) -> nn.Module:
    m = fn(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


class TinyGPT(nn.Module):
    def __init__(self, vocab: int = 8192, d_model: int = 512, n_layer: int = 8,
                 n_head: int = 8, seq_len: int = 256) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_head, 4 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.register_buffer("causal", nn.Transformer.generate_square_subsequent_mask(seq_len),
                             persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        h = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        h = self.blocks(h, mask=self.causal[:t, :t], is_causal=True)
        return self.head(self.norm(h))


class CNNTiny(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(name: str, seq_len: int = 256) -> tuple[nn.Module, str]:
    """Return (model, task) where task is 'image' or 'lm'."""
    if name in ("resnet18", "resnet50"):
        import torchvision.models as tvm

        return _cifar_resnet(getattr(tvm, name), 10), "image"
    if name == "transformer_small":
        return TinyGPT(seq_len=seq_len), "lm"
    if name == "cnn_tiny":
        return CNNTiny(), "image"
    raise ValueError(f"unknown model {name!r}")


def param_stats(model: nn.Module) -> dict:
    ps = [p for p in model.parameters() if p.requires_grad]
    return {"params": sum(p.numel() for p in ps),
            "grad_bytes_fp32": sum(p.numel() * 4 for p in ps),
            "tensors": len(ps)}
