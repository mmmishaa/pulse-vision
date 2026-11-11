import pytest
import torch
from src.models.losses import NegativePearsonCorrelation

def test_pearson_loss_math():
    loss = NegativePearsonCorrelation()
    pred = torch.randn(size=(16, 100))
    target = pred.clone()

    assert loss(pred, target) == 0.0

    pred = pred * 100

    assert loss(pred, target) == 0.0

    pred = pred * -1

    assert loss(pred, target) == 2.0

    pred = pred + 500

    assert loss(pred, target) == 2.0

def test_backward_gradients():
    loss = NegativePearsonCorrelation()
    pred = torch.randn(size=(16, 100), requires_grad=True)
    target = torch.randn(size=(16, 100))

    res_loss = loss(pred, target)
    res_loss.backward()

    assert pred.grad is not None and not torch.isnan(pred.grad).any()