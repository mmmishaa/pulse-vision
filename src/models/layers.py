import torch
import torch.nn as nn

class TemporalShiftModule(nn.Module):

    def __init__(self, n_segment: int, ch_splitter: int = 3):

        super().__init__()

        self.n_segment = n_segment

        if ch_splitter < 3:
            raise ValueError("Channel divider error; each part is at least one-third")

        self.ch_splitter = ch_splitter

    def forward(self, x: torch.Tensor):
        """
        x: input (B * N, C, H, W) -> TSM (B, N, C, H, W) -> output (B * N, C, H, W)
        """

        if x.ndim != 4:
            raise ValueError("Incorrect input tensor dimension")
        
        C, H, W = x.shape[1:]
        x = torch.reshape(input=x, shape=(-1, self.n_segment, C, H, W))

        ch_sep = C // self.ch_splitter
        C_past, C_future, C_present = ch_sep, ch_sep, C - 2 * ch_sep

        tsm_x = torch.zeros_like(x)

        tsm_x[:, 1:, :C_past, ...] = x[:, :-1, :C_past, ...] # past
        tsm_x[:, :-1, C_past:C_past + C_future, ...] = x[:, 1:, C_past:C_past + C_future, ...] # future
        tsm_x[:, :, C_past + C_future:, ...] = x[:, :, C_past + C_future:, ...] # present

        tsm_x = torch.reshape(input=tsm_x, shape=(-1, C, H, W))

        return tsm_x
