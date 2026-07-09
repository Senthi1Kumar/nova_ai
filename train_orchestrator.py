# train_orchestrator.py  — run on your laptop, not on Jetson
"""
1. Generate training data by running SmolLM2 on your auth event log
2. Train MLP on (context_vector, weights) pairs
3. Export to ONNX
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

class PolicyMLPTorch(nn.Module):
    def __init__(self, in_dim=20, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.LayerNorm(hidden), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden, 32),
            nn.LayerNorm(32),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(32, 6),
        )

    def forward(self, x):
        out = self.net(x)
        weights     = torch.softmax(out[:, :4], dim=-1)   # sum to 1
        uncertainty = torch.sigmoid(out[:, 4:5])           # 0–1
        thresh_d    = torch.tanh(out[:, 5:6]) * 0.15      # ±0.15
        return torch.cat([weights, uncertainty, thresh_d], dim=-1)


def export_to_onnx(model: PolicyMLPTorch, path: str):
    model.eval()
    dummy = torch.zeros(1, 20)
    torch.onnx.export(
        model, (dummy,), path,
        input_names=["context"],
        output_names=["output"],
        dynamic_axes={"context": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported to {path} ({Path(path).stat().st_size / 1024:.0f} KB)")