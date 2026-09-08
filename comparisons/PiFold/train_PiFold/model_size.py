"""Print PiFold's parameter count and approximate memory footprint.

Usage:
    python model_size.py [same CLI args as main.py, e.g. --hidden_dim 128]
"""
import sys
import types

try:
    import torch_scatter  # noqa: F401
except ImportError:
    # torch_scatter is only exercised at forward time, not during __init__,
    # so a stub is enough to build the model and count parameters.
    stub = types.ModuleType("torch_scatter")

    def _unimplemented(*args, **kwargs):
        raise NotImplementedError("torch_scatter is not installed locally")

    stub.scatter_sum = _unimplemented
    stub.scatter_softmax = _unimplemented
    stub.scatter_mean = _unimplemented
    sys.modules["torch_scatter"] = stub

from parser import create_parser
from methods.prodesign_model import ProDesign_Model


def main():
    args = create_parser()
    model = ProDesign_Model(args)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params:     {total:,} (~{total / 1e6:.2f}M)")
    print(f"Trainable params: {trainable:,} (~{trainable / 1e6:.2f}M)")
    print(f"Approx size fp32: {total * 4 / 1024**2:.2f} MB")
    print(f"Approx size fp16: {total * 2 / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
