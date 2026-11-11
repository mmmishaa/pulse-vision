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

class AttentionMask(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        x: input (B * N, C, H, W) -> output (B * N, C, H, W)
        """

        raw_attention_mask = self.conv(x) # (B * N, 1, H, W)
        attention_mask = self.sigmoid(raw_attention_mask) # (B * N, 1, H, W)

        H, W = x.shape[2:]
        hw = H * W

        eps = 1e-6

        attention_mask = attention_mask * (hw / (eps + 2 * torch.norm(attention_mask, p=1, dim=(1, 2, 3), keepdim=True)))

        masked_x = attention_mask * x

        return masked_x
