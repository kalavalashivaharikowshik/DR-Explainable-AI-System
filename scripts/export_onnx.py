"""Export the trained PyTorch model to ONNX for light-weight serving.

    python scripts/export_onnx.py                # models/trained/best_model.pth -> .onnx

The ONNX graph has two outputs:
  logits    [1, 5]
  features  [1, 1536, h, w]  output of the last convolutional block

For EfficientNet the head is  global-average-pool -> dropout -> linear, so the
gradient of class c with respect to the features is constant: w_c / (h*w).
Grad-CAM therefore equals ReLU(sum_k w_ck * A_k) up to a positive scale, which
the server computes in NumPy from `features` and the classifier weights stored
next to the model (best_model_head.npz). No autograd is needed at inference.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.config import MODEL_INPUT_SIZE, MODEL_PATH  # noqa: E402
from src.models.classification.efficientnet_model import get_model  # noqa: E402


class WithFeatures(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.features(x)
        pooled = torch.flatten(self.model.avgpool(features), 1)
        logits = self.model.classifier(pooled)
        return logits, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=MODEL_PATH)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or os.path.splitext(args.weights)[0] + ".onnx"
    head_path = os.path.splitext(output)[0] + "_head.npz"

    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    torch.onnx.export(
        WithFeatures(model), dummy, output,
        input_names=["input"], output_names=["logits", "features"],
        opset_version=17, dynamo=False,
    )

    linear = model.classifier[1]
    np.savez(head_path,
             weight=linear.weight.detach().numpy().astype(np.float32),
             bias=linear.bias.detach().numpy().astype(np.float32))

    print(f"Saved {output} ({os.path.getsize(output) / 1e6:.1f} MB) and {head_path}")


if __name__ == "__main__":
    main()
