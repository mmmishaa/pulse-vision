import pytest
import torch
from src.models.layers import AttentionMask

def test_output_shape():
    att_mask = AttentionMask(in_channels=32)
    x = torch.randn(size=(15, 32, 72, 72))
    masked_x = att_mask(x)

    assert masked_x.shape == (15, 32, 72, 72)

def test_normalization_on_uniform_input():
    att_mask = AttentionMask(in_channels=32)
    x = torch.ones(size=(15, 32, 72, 72))
    test_x = torch.full_like(x, fill_value=0.5)
    masked_x = att_mask(x)

    assert torch.allclose(masked_x, test_x, atol=1e-4)

def test_frame_independence():
    att_mask = AttentionMask(in_channels=32)
    x = torch.randn(size=(15, 32, 72, 72)) * 100
    masked_x0_1 = att_mask(x)[:1, ...]

    x0 = x[:1, ...]
    masked_x0_2 = att_mask(x0)

    assert torch.allclose(masked_x0_1, masked_x0_2)

def test_backward_gradients():
    att_mask = AttentionMask(in_channels=32)
    x = torch.randn(size=(15, 32, 72, 72), requires_grad=True)
    out = att_mask(x)
    s = out.sum()
    s.backward()

    assert x.grad is not None and not torch.isnan(x.grad).any()
    assert not torch.isnan(att_mask.conv.weight.grad).any()