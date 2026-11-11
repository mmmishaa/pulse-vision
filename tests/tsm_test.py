import pytest
import torch
from src.models.layers import TemporalShiftModule

def test_output_shape():
    tsm = TemporalShiftModule(n_segment=5, ch_splitter=3)

    x = torch.randn((15, 3, 72, 72))

    tsm_x = tsm(x)

    assert tsm_x.shape == (15, 3, 72, 72)

def test_shift_and_batch_isolation():
    tsm = TemporalShiftModule(n_segment=3, ch_splitter=3)

    x = torch.arange(start=1, end=97).reshape(shape=(6, 4, 2, 2))
    tsm_x = tsm(x)

    expected = torch.tensor([
        [
            [[0, 0], [0, 0]],
            [[21, 22], [23, 24]],
            [[9, 10], [11, 12]],
            [[13, 14], [15, 16]],
        ],
        [
            [[1, 2], [3, 4]],
            [[37, 38], [39, 40]],
            [[25, 26], [27, 28]],
            [[29, 30], [31, 32]],
        ],
        [
            [[17, 18], [19, 20]],
            [[0, 0], [0, 0]],
            [[41, 42], [43, 44]],
            [[45, 46], [47, 48]],
        ],
        [
            [[0, 0], [0, 0]],
            [[69, 70], [71, 72]],
            [[57, 58], [59, 60]],
            [[61, 62], [63, 64]],
        ],
        [
            [[49, 50], [51, 52]],
            [[85, 86], [87, 88]],
            [[73, 74], [75, 76]],
            [[77, 78], [79, 80]],
        ],
        [
            [[65, 66], [67, 68]],
            [[0, 0], [0, 0]],
            [[89, 90], [91, 92]],
            [[93, 94], [95, 96]],
        ],
    ])

    assert torch.equal(tsm_x, expected)