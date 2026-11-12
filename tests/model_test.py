import pytest
import torch

from src.models.model import EfficientPhys_C
from src.models.losses import NegativePearsonCorrelation

def test_output_shape():
    x = torch.randn(size=(16, 10, 3, 72, 72))

    model = EfficientPhys_C(n_frames=10)

    out = model(x)

    assert out.shape == (16, 10)

def test_backward_gradients():
    x = torch.randn(size=(2, 10, 3, 72, 72), requires_grad=True)
    target = torch.randn(size=(2, 10))

    model = EfficientPhys_C(n_frames=10)
    loss = NegativePearsonCorrelation()

    out = model(x)
    loss_val = loss(out, target)
    loss_val.backward()

    assert x.grad is not None and not torch.isnan(x.grad).any()
    assert all(param.grad is not None and not torch.isnan(param.grad).any() and param.grad.abs().sum() > 0 for param in model.parameters())