import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    SinusoidalPositionEmbeddings,
    ResidualBlock3D,
    SelfAttention3D,
    Downsample3D,
    Upsample3D,
)


class UNet3D(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        time_emb_dim = config.model_channels * 4

        # base time embedding (sin/cos + 2-layer MLP)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.model_channels),
            nn.Linear(config.model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # optional categorical conditioning (size class etc.) — added to time embedding
        self.num_size_classes = getattr(config, "num_size_classes", 0)
        if self.num_size_classes > 0:
            self.size_embed = nn.Embedding(self.num_size_classes + 1, time_emb_dim)
            self.null_class = self.num_size_classes   # last index reserved for "no condition"

        # optional spatial conditioning (e.g. seg map) — concatenated channel-wise to x
        self.cond_channels = getattr(config, "cond_channels", 0)
        self.init_conv = nn.Conv3d(config.in_channels + self.cond_channels,
                                   config.model_channels, 3, padding=1)

        channels = [config.model_channels * m for m in config.channel_mult]
        skip_channels = [config.model_channels]
        resolution = config.image_size
        in_ch = config.model_channels

        # ----- down path -----
        self.downs = nn.ModuleList()
        for i, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                blocks.append(ResidualBlock3D(in_ch, out_ch, time_emb_dim,
                                              config.num_groups, config.dropout))
                in_ch = out_ch
                skip_channels.append(out_ch)
            if resolution in config.attention_resolutions:
                blocks.append(SelfAttention3D(out_ch, num_groups=config.num_groups))
            if i < len(channels) - 1:
                blocks.append(Downsample3D(out_ch))
                skip_channels.append(out_ch)
                resolution //= 2
            self.downs.append(blocks)

        # ----- bottleneck -----
        mid_ch = channels[-1]
        self.mid_block1 = ResidualBlock3D(mid_ch, mid_ch, time_emb_dim,
                                          config.num_groups, config.dropout)
        self.mid_attn = SelfAttention3D(mid_ch, num_groups=config.num_groups)
        self.mid_block2 = ResidualBlock3D(mid_ch, mid_ch, time_emb_dim,
                                          config.num_groups, config.dropout)

        # ----- up path -----
        self.ups = nn.ModuleList()
        for i, out_ch in enumerate(reversed(channels)):
            blocks = nn.ModuleList()
            for _ in range(config.num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                blocks.append(ResidualBlock3D(in_ch + skip_ch, out_ch, time_emb_dim,
                                              config.num_groups, config.dropout))
                in_ch = out_ch
            if resolution in config.attention_resolutions:
                blocks.append(SelfAttention3D(out_ch, num_groups=config.num_groups))
            if i < len(channels) - 1:
                blocks.append(Upsample3D(out_ch))
                resolution *= 2
            self.ups.append(blocks)

        self.final_norm = nn.GroupNorm(config.num_groups, config.model_channels)
        self.final_conv = nn.Conv3d(config.model_channels, config.in_channels, 3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, t, cond=None, size_cond=None):
        t_emb = self.time_mlp(t)

        # add size-class embedding (uses the null index if size_cond not given)
        if self.num_size_classes > 0:
            if size_cond is None:
                size_cond = torch.full((x.shape[0],), self.null_class,
                                       device=x.device, dtype=torch.long)
            t_emb = t_emb + self.size_embed(size_cond)

        # concat spatial conditioning (uses zeros if cond not given)
        if self.cond_channels > 0:
            if cond is None:
                cond = torch.zeros((x.shape[0], self.cond_channels, *x.shape[2:]),
                                   device=x.device)
            x = torch.cat([x, cond], dim=1)

        x = self.init_conv(x)
        skips = [x]

        for blocks in self.downs:
            for block in blocks:
                if isinstance(block, ResidualBlock3D):
                    x = block(x, t_emb)
                    skips.append(x)
                elif isinstance(block, Downsample3D):
                    x = block(x)
                    skips.append(x)
                else:
                    x = block(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for blocks in self.ups:
            for block in blocks:
                if isinstance(block, ResidualBlock3D):
                    x = torch.cat([x, skips.pop()], dim=1)
                    x = block(x, t_emb)
                else:
                    x = block(x)

        return self.final_conv(F.silu(self.final_norm(x)))
