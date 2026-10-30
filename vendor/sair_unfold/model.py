import math
import torch
from torch import nn
from torch.nn import functional as F


def normalization(channels):
    groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


class TimeEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 4), nn.SiLU(),
                                 nn.Linear(channels * 4, channels * 4))

    def forward(self, t):
        half = self.channels // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(
            half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
        phases = t.float()[:, None] * frequencies[None]
        embedding = torch.cat([phases.cos(), phases.sin()], dim=1)
        if self.channels % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, time_channels):
        super().__init__()
        self.in_layers = nn.Sequential(normalization(input_channels), nn.SiLU(),
                                       nn.Conv2d(input_channels, output_channels, 3, padding=1))
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, output_channels))
        self.out_layers = nn.Sequential(normalization(output_channels), nn.SiLU(),
                                        nn.Conv2d(output_channels, output_channels, 3, padding=1))
        self.skip = (nn.Identity() if input_channels == output_channels
                     else nn.Conv2d(input_channels, output_channels, 1))

    def forward(self, x, time):
        h = self.in_layers(x) + self.time(time)[:, :, None, None]
        return self.skip(x) + self.out_layers(h)


class UNet(nn.Module):
    """Single-channel epsilon predictor; no visibility conditioning."""
    def __init__(self, base_channels=32, channel_mult=(1, 2, 4, 4)):
        super().__init__()
        channels = [base_channels * m for m in channel_mult]
        time_channels = base_channels * 4
        self.time = TimeEmbedding(base_channels)
        self.input = nn.Conv2d(1, channels[0], 3, padding=1)
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, c in enumerate(channels):
            self.encoder.append(nn.ModuleList([ResBlock(c, c, time_channels),
                                               ResBlock(c, c, time_channels)]))
            if i + 1 < len(channels):
                self.down.append(nn.Conv2d(c, channels[i + 1], 3, stride=2, padding=1))
        self.middle = nn.ModuleList([ResBlock(channels[-1], channels[-1], time_channels),
                                    ResBlock(channels[-1], channels[-1], time_channels)])
        self.decoder = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in reversed(range(len(channels))):
            c = channels[i]
            self.decoder.append(nn.ModuleList([ResBlock(c * 2, c, time_channels),
                                               ResBlock(c, c, time_channels)]))
            if i > 0:
                self.up.append(nn.Conv2d(c, channels[i - 1], 3, padding=1))
        self.output = nn.Sequential(normalization(channels[0]), nn.SiLU(),
                                    nn.Conv2d(channels[0], 1, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x, t):
        time = self.time(t)
        h, skips = self.input(x), []
        for i, blocks in enumerate(self.encoder):
            for block in blocks:
                h = block(h, time)
            skips.append(h)
            if i < len(self.down):
                h = self.down[i](h)
        for block in self.middle:
            h = block(h, time)
        for i, blocks in enumerate(self.decoder):
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, time)
            if i < len(self.up):
                h = self.up[i](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.output(h)
