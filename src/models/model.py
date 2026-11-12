import torch
import torch.nn as nn
from src.models.layers import TemporalShiftModule, AttentionMask

class EfficientPhys_C(nn.Module):

    def __init__(self, n_frames):

        super().__init__()

        self.n_frames = n_frames

        self.batchnorm = nn.BatchNorm2d(num_features=3)

        self.block1 = nn.Sequential(
            TemporalShiftModule(n_segment=self.n_frames, ch_splitter=3),
            nn.Conv2d(in_channels=3, out_channels=32, padding=1, kernel_size=3, stride=1),
            nn.Tanh(),
            TemporalShiftModule(n_segment=self.n_frames, ch_splitter=3),
            nn.Conv2d(in_channels=32, out_channels=32, padding=1, kernel_size=3, stride=1),
            nn.Tanh(),
            AttentionMask(in_channels=32)
        ) # (B * N, 3, 72, 72) -> (B * N, 32, 72, 72)

        self.downsample_block1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        ) # (B * N, 32, 72, 72) -> (B * N, 32, 36, 36)

        self.block2 = nn.Sequential(
            TemporalShiftModule(n_segment=self.n_frames, ch_splitter=3),
            nn.Conv2d(in_channels=32, out_channels=64, padding=1, kernel_size=3, stride=1),
            nn.Tanh(),
            TemporalShiftModule(n_segment=self.n_frames, ch_splitter=3),
            nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1),
            nn.Tanh(),
            AttentionMask(in_channels=64)
        ) # (B * N, 32, 36, 36) -> (B * N, 64, 36, 36)

        self.downsample_block2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        ) # (B * N, 64, 36, 36) -> (B * N, 64, 18, 18)

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=64*18*18, out_features=128),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=1, bias=False)
        ) # (B * N, 64, 18, 18) -> (B * N, 64 * 18 * 18) -> (B * N, 128) -> (B * N, 128) -> (B * N, 1)

    def forward(self, x: torch.Tensor):
        """
        x: input (B, N, C, H, W) = (B, N, 3, 72, 72)
        output: (B, N)
        """
        # diff layer
        x = torch.diff(x, dim=1) # (B, N - 1, C, H, W)
        x = torch.cat((x, x[:, -1:, ...]), dim=1) # (B, N, C, H, W)

        C, H, W = x.shape[2:]
        x = x.reshape(shape=(-1, C, H, W)) # (B * N, C, H, W)

        # batchnorm layer
        x = self.batchnorm(x)

        out = self.block1(x)
        out = self.downsample_block1(out)

        out = self.block2(out)
        out = self.downsample_block2(out)

        out = self.head(out)

        out = out.reshape(shape=(-1, self.n_frames))

        return out
