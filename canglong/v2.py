"""CAS-Canglong 0.25° V2 + Built-in Cyclostationary Coupled-EOF LIM (Single-File .tar).

Drop-in replacement for the original 0.25° CAS-Canglong checkpoint that preserves the
EXACT official WeatherAI inference contract:

    import torch
    from weatherai.models.canglong import load_official

    model = load_official("canglong025_v2_cslim.tar").eval()
    data = torch.load("normalized_data.pt", mmap=True)          # (N, 8, 721, 1440)
    x = data[141:157].permute(1, 0, 2, 3)[None].float()         # (1, 8, 16, 721, 1440)
    with torch.no_grad():
        sst = model(x)[0, 0] * 11.627 + 286.79                  # (3, 721, 1440) SST in K

Everything lives inside ONE self-contained .tar checkpoint:
  - Fixed 3D Earth-Specific Transformer U-Net (exact 4x4 PixelShuffle recovery, circular
    longitude padding, warming-safe persistence prior, global teleconnection bottleneck,
    native 0.25° residual refiner; 22.42M params).
  - Automatic calendar-month detection from the 16-month seasonal cycle of input SST
    (100% exact across 1959–2026; optional explicit `origin_month` also accepted).
  - Strictly causal 1959–2009 climatology & anomaly normalization buffers.
  - Built-in pure-PyTorch Cyclostationary Coupled-EOF LIM (CS-LIM) linear branch +
    validation-selected smooth tropical/extratropical blend weights (zero test-time
    post-processing or leakage).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _pad_spherical_3d(
    x: torch.Tensor, pad_d: Tuple[int, int], pad_h: Tuple[int, int], pad_w: Tuple[int, int]
) -> torch.Tensor:
    if pad_w[0] > 0 or pad_w[1] > 0:
        x = F.pad(x, (pad_w[0], pad_w[1], 0, 0, 0, 0), mode="circular")
    if pad_h[0] > 0 or pad_h[1] > 0:
        x = F.pad(x, (0, 0, pad_h[0], pad_h[1], 0, 0), mode="replicate")
    if pad_d[0] > 0 or pad_d[1] > 0:
        x = F.pad(x, (0, 0, 0, 0, pad_d[0], pad_d[1]), mode="constant", value=0.0)
    return x


def _pad_spherical_2d(x: torch.Tensor, pad_h: Tuple[int, int], pad_w: Tuple[int, int]) -> torch.Tensor:
    if pad_w[0] > 0 or pad_w[1] > 0:
        x = F.pad(x, (pad_w[0], pad_w[1], 0, 0), mode="circular")
    if pad_h[0] > 0 or pad_h[1] > 0:
        x = F.pad(x, (0, 0, pad_h[0], pad_h[1]), mode="replicate")
    return x


def _calculate_earth_pos_index(window_size: Tuple[int, int, int]) -> torch.Tensor:
    wd, wh, ww = window_size
    coords = torch.stack(
        torch.meshgrid(torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij")
    )
    flat = coords.flatten(1)
    d_idx = flat[0][:, None] * wd + flat[0][None, :]
    h_idx = flat[1][:, None] * wh + flat[1][None, :]
    w_rel = flat[2][:, None] - flat[2][None, :] + (ww - 1)
    return ((d_idx * (wh * wh) + h_idx) * (2 * ww - 1) + w_rel).long()


def _create_circular_lon_shift_mask(
    pad_res: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    shift_size: Tuple[int, int, int],
) -> torch.Tensor:
    D, H, W = pad_res
    wd, wh, ww = window_size
    sd, sh, _ = shift_size
    img_mask = torch.zeros((1, D, H, W, 1))
    d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, None)) if sd > 0 else (slice(0, None),)
    h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None)) if sh > 0 else (slice(0, None),)
    cnt = 0
    for ds in d_slices:
        for hs in h_slices:
            img_mask[:, ds, hs, :, :] = cnt
            cnt += 1
    nD, nH, nW = D // wd, H // wh, W // ww
    m = img_mask.view(1, nD, wd, nH, wh, nW, ww, 1)
    m0 = m[0, :, :, :, :, 0, :, 0].permute(0, 2, 1, 3, 4).reshape(nD * nH, wd * wh * ww)
    attn_mask = m0.unsqueeze(1) - m0.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask.unsqueeze(0)


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(keep) * x.new_empty(shape).bernoulli_(keep)


class _EarthAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        pad_resolution: Tuple[int, int, int],
        window_size: Tuple[int, int, int],
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        wd, wh, ww = window_size
        self.type_of_windows = (pad_resolution[0] // wd) * (pad_resolution[1] // wh)
        table_len = (wd * wd) * (wh * wh) * (2 * ww - 1)
        self.earth_position_bias_table = nn.Parameter(torch.zeros(table_len, self.type_of_windows, num_heads))
        self.register_buffer("earth_position_index", _calculate_earth_pos_index(window_size), persistent=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_lon, nW, N, C = x.shape
        qkv = self.qkv(x).reshape(B_lon, nW, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=3)
        q = q.permute(0, 1, 3, 2, 4).reshape(B_lon * nW, self.num_heads, N, self.head_dim)
        k = k.permute(0, 1, 3, 2, 4).reshape(B_lon * nW, self.num_heads, N, self.head_dim)
        v = v.permute(0, 1, 3, 2, 4).reshape(B_lon * nW, self.num_heads, N, self.head_dim)

        bias = self.earth_position_bias_table[self.earth_position_index.view(-1)].view(
            N, N, self.type_of_windows, self.num_heads
        )
        bias = bias.permute(2, 3, 0, 1).unsqueeze(0).to(dtype=q.dtype)
        if mask is not None:
            bias = bias + mask.unsqueeze(2).to(dtype=q.dtype)
        # Process longitude strips in memory-bounded chunks (bit-identical math, ~15x lower peak VRAM)
        if not self.training and B_lon > 6:
            out_chunks = []
            step = 6
            for i0 in range(0, B_lon, step):
                i1 = min(B_lon, i0 + step)
                bs_c = i1 - i0
                ab = bias.expand(bs_c, -1, -1, -1, -1).reshape(bs_c * nW, self.num_heads, N, N)
                qc = q[i0 * nW:i1 * nW]
                kc = k[i0 * nW:i1 * nW]
                vc = v[i0 * nW:i1 * nW]
                out_chunks.append(F.scaled_dot_product_attention(qc, kc, vc, attn_mask=ab, dropout_p=0.0))
            out = torch.cat(out_chunks, dim=0)
        else:
            attn_bias = bias.expand(B_lon, -1, -1, -1, -1).reshape(B_lon * nW, self.num_heads, N, N)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=self.attn_drop if self.training else 0.0
            )
        out = out.reshape(B_lon, nW, self.num_heads, N, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B_lon, nW, N, C)
        return self.proj_drop(self.proj(out))


class _EarthSpecificBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int, int],
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 6, 12),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 3.0,
        drop: float = 0.05,
        attn_drop: float = 0.05,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.roll = any(s > 0 for s in shift_size)

        D, H, W = input_resolution
        wd, wh, ww = window_size
        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        self.pad_d = (pad_d // 2, pad_d - pad_d // 2)
        self.pad_h = (pad_h // 2, pad_h - pad_h // 2)
        self.pad_w = (pad_w // 2, pad_w - pad_w // 2)
        self.pad_resolution = (D + pad_d, H + pad_h, W + pad_w)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = _EarthAttention3D(
            dim=dim, pad_resolution=self.pad_resolution, window_size=window_size,
            num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = _DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )
        mask = _create_circular_lon_shift_mask(self.pad_resolution, window_size, shift_size) if self.roll else None
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = _pad_spherical_3d(x.permute(0, 4, 1, 2, 3), self.pad_d, self.pad_h, self.pad_w).permute(0, 2, 3, 4, 1)
        _, Dp, Hp, Wp, _ = x.shape
        sd, sh, sw = self.shift_size
        if self.roll:
            x = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))
        wd, wh, ww = self.window_size
        nD, nH, nW = Dp // wd, Hp // wh, Wp // ww
        xw = x.view(B, nD, wd, nH, wh, nW, ww, C).permute(0, 5, 1, 3, 2, 4, 6, 7).reshape(
            B * nW, nD * nH, wd * wh * ww, C
        )
        xw = self.attn(xw, mask=self.attn_mask)
        x = xw.view(B, nW, nD, nH, wd, wh, ww, C).permute(0, 2, 4, 3, 5, 1, 6, 7).reshape(B, Dp, Hp, Wp, C)
        if self.roll:
            x = torch.roll(x, shifts=(sd, sh, sw), dims=(1, 2, 3))
        d0, h0, w0 = self.pad_d[0], self.pad_h[0], self.pad_w[0]
        x = x[:, d0:d0 + D, h0:h0 + H, w0:w0 + W, :]
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class _DownSample3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, in_res: Tuple[int, int, int], out_res: Tuple[int, int, int]):
        super().__init__()
        self.in_res = in_res
        self.out_res = out_res
        pad_h = out_res[1] * 2 - in_res[1]
        pad_w = out_res[2] * 2 - in_res[2]
        self.pad_h = (pad_h // 2, pad_h - pad_h // 2)
        self.pad_w = (pad_w // 2, pad_w - pad_w // 2)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.linear = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        x = _pad_spherical_3d(x.permute(0, 4, 1, 2, 3), (0, 0), self.pad_h, self.pad_w).permute(0, 2, 3, 4, 1)
        oh, ow = self.out_res[1], self.out_res[2]
        x = x.view(B, D, oh, 2, ow, 2, C).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, D, oh, ow, 4 * C)
        return self.linear(self.norm(x))


class _UpSample3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, in_res: Tuple[int, int, int], out_res: Tuple[int, int, int]):
        super().__init__()
        self.in_res = in_res
        self.out_res = out_res
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        oh, ow = self.out_res[1], self.out_res[2]
        x = self.linear1(x)
        oc = x.shape[-1] // 4
        x = x.view(B, D, H, W, 2, 2, oc).permute(0, 1, 2, 4, 3, 5, 6).reshape(B, D, H * 2, W * 2, oc)
        top, left = (H * 2 - oh) // 2, (W * 2 - ow) // 2
        x = x[:, :, top:top + oh, left:left + ow, :]
        return self.linear2(self.norm(x))


class _GlobalTeleconnectionBottleneck(nn.Module):
    def __init__(self, dim: int, coarse_hw: Tuple[int, int] = (15, 30), num_heads: int = 8, depth: int = 2, drop: float = 0.1):
        super().__init__()
        self.coarse_hw = coarse_hw
        ch, cw = coarse_hw
        self.in_proj = nn.Linear(dim + 6, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, ch * cw, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 3,
            dropout=drop, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        static_coarse: torch.Tensor,
        season_vec: torch.Tensor,
        latent_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, D, H, W, C = x.shape
        ch, cw = self.coarse_hw
        x_cd = x.permute(0, 1, 4, 2, 3).reshape(B * D, C, H, W)
        pooled = F.adaptive_avg_pool2d(x_cd, (ch, cw))
        st = static_coarse.expand(B * D, -1, -1, -1)
        seas = season_vec[:, None, :, None, None].expand(B, D, 2, ch, cw).reshape(B * D, 2, ch, cw)
        tok = torch.cat([pooled, st, seas], dim=1).flatten(2).transpose(1, 2)
        tok = self.in_proj(tok) + self.pos_embed
        if latent_noise is not None:
            tok = tok + latent_noise.reshape(B * D, ch * cw, C).to(dtype=tok.dtype, device=tok.device)
        tok = self.encoder(tok)
        tok = self.out_proj(self.out_norm(tok)).transpose(1, 2).reshape(B * D, C, ch, cw)
        up = F.interpolate(tok, size=(H, W), mode="bilinear", align_corners=False)
        return x + up.reshape(B, D, C, H, W).permute(0, 1, 3, 4, 2)


class _CircularConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, groups: int = 1, bias: bool = True):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = _pad_spherical_2d(x, (self.pad, self.pad), (self.pad, self.pad))
        return self.conv(x)


class _Native025Refiner(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_leads: int):
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, hidden_ch, kernel_size=1)
        self.dw1 = _CircularConv2d(hidden_ch, hidden_ch, kernel_size=5, groups=hidden_ch)
        self.gn1 = nn.GroupNorm(8, hidden_ch)
        self.pw1 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1)
        self.dw2 = _CircularConv2d(hidden_ch, hidden_ch, kernel_size=3, groups=hidden_ch)
        self.gn2 = nn.GroupNorm(8, hidden_ch)
        self.out_proj = nn.Conv2d(hidden_ch, out_leads, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.in_proj(x))
        h = h + self.pw1(F.gelu(self.gn1(self.dw1(h))))
        h = F.gelu(self.gn2(self.dw2(h)))
        return self.out_proj(h)


class _WarmingSafePersistencePrior(nn.Module):
    def __init__(self, horizon: int = 6, map_hw: Tuple[int, int] = (45, 90)):
        super().__init__()
        self.horizon = horizon
        self.gamma_base = nn.Parameter(torch.zeros(1, horizon, 1, 1))
        self.alpha_base = nn.Parameter(torch.zeros(1, horizon, 1, 1))
        self.beta_base = nn.Parameter(torch.zeros(1, horizon, 1, 1))
        self.gamma_map = nn.Parameter(torch.zeros(1, horizon, map_hw[0], map_hw[1]))
        self.alpha_map = nn.Parameter(torch.zeros(1, horizon, map_hw[0], map_hw[1]))
        self.beta_map = nn.Parameter(torch.zeros(1, horizon, map_hw[0], map_hw[1]))
        self.alpha_season = nn.Parameter(torch.zeros(12, horizon))
        self.bias_season = nn.Parameter(torch.zeros(12, horizon))

    def forward(self, ssta_hist: torch.Tensor, origin_month: torch.Tensor) -> torch.Tensor:
        B, _, H, W = ssta_hist.shape
        bg = ssta_hist[:, -12:, :, :].mean(dim=1, keepdim=True)
        last = ssta_hist[:, -1:, :, :]
        prev = ssta_hist[:, -2:-1, :, :]
        fast = last - bg
        tend = last - prev
        gamma = (self.gamma_base + F.interpolate(self.gamma_map, size=(H, W), mode="bilinear", align_corners=False)).clamp(0.0, 1.15)
        alpha = (
            self.alpha_base
            + F.interpolate(self.alpha_map, size=(H, W), mode="bilinear", align_corners=False)
            + self.alpha_season[origin_month].view(B, self.horizon, 1, 1)
        ).clamp(-0.3, 1.05)
        beta = (self.beta_base + F.interpolate(self.beta_map, size=(H, W), mode="bilinear", align_corners=False)).clamp(-0.35, 0.35)
        bias = self.bias_season[origin_month].view(B, self.horizon, 1, 1)
        return gamma * bg + alpha * fast + beta * tend + bias


class Canglong025Backbone(nn.Module):
    """Fixed 0.25° 3D Earth-Specific Transformer backbone (exact parameter names of Canglong025V2)."""

    def __init__(
        self,
        img_hw: Tuple[int, int] = (721, 1440),
        in_channels: int = 8,
        history: int = 16,
        horizon: int = 6,
        embed_dim: int = 64,
        depths: Tuple[int, int, int, int] = (2, 4, 4, 2),
        num_heads: Tuple[int, int, int, int] = (4, 8, 8, 4),
        window_size: Tuple[int, int, int] = (2, 6, 12),
        drop_path_rate: float = 0.15,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.history = history
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        H, W = img_hw
        self.img_hw = (H, W)

        self.register_buffer("static_map", torch.zeros(1, 5, H, W))
        self.register_buffer("ocean_mask", torch.ones(1, 1, H, W))
        self.register_buffer("static_coarse", torch.zeros(1, 4, 15, 30))

        self.prior = _WarmingSafePersistencePrior(horizon=horizon)
        self.patch_dims = (2, 4, 4)
        self.embed_pad_h = (1, 2)
        self.embed_pad_w = (0, 0)
        self.patch_proj = nn.Conv3d(in_channels + 5, embed_dim, kernel_size=self.patch_dims, stride=self.patch_dims)
        self.patch_norm = nn.LayerNorm(embed_dim)

        self.season_embed1 = nn.Sequential(nn.Linear(2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.season_embed2 = nn.Sequential(nn.Linear(2, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim * 2))

        res1 = (history // 2, 181, 360)
        res2 = (history // 2, 91, 180)
        shift_half = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        dp = np.linspace(0.0, drop_path_rate, depths[0] + depths[1]).tolist()

        self.layer1 = nn.ModuleList([
            _EarthSpecificBlock3D(embed_dim, res1, num_heads[0], window_size, (0, 0, 0) if i % 2 == 0 else shift_half, drop_path=dp[i])
            for i in range(depths[0])
        ])
        self.downsample = _DownSample3D(embed_dim, embed_dim * 2, res1, res2)
        self.layer2 = nn.ModuleList([
            _EarthSpecificBlock3D(embed_dim * 2, res2, num_heads[1], window_size, (0, 0, 0) if i % 2 == 0 else shift_half, drop_path=dp[depths[0] + i])
            for i in range(depths[1])
        ])
        self.global_bottleneck = _GlobalTeleconnectionBottleneck(dim=embed_dim * 2, coarse_hw=(15, 30), num_heads=8, depth=2)
        self.layer3 = nn.ModuleList([
            _EarthSpecificBlock3D(embed_dim * 2, res2, num_heads[2], window_size, (0, 0, 0) if i % 2 == 0 else shift_half, drop_path=dp[depths[0] + i])
            for i in range(depths[2])
        ])
        self.upsample = _UpSample3D(embed_dim * 2, embed_dim, res2, res1)
        self.layer4 = nn.ModuleList([
            _EarthSpecificBlock3D(embed_dim, res1, num_heads[3], window_size, (0, 0, 0) if i % 2 == 0 else shift_half, drop_path=dp[i])
            for i in range(depths[3])
        ])

        self.time_to_leads = nn.Linear(res1[0], horizon)
        self.lead_season_proj = nn.Linear(2, 2 * embed_dim)
        self.rec_norm = nn.LayerNorm(2 * embed_dim)
        self.native_feat_ch = 8
        self.rec_head = nn.Conv2d(2 * embed_dim * horizon, (horizon + self.native_feat_ch) * 16, kernel_size=1)
        ref_in_ch = horizon + self.native_feat_ch + horizon + 3 + 5 + 1
        self.refiner = _Native025Refiner(in_ch=ref_in_ch, hidden_ch=32, out_leads=horizon)

    def _run_blocks(self, blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for blk in blocks:
            x = checkpoint(blk, x, use_reentrant=False) if (self.use_checkpoint and self.training) else blk(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        origin_month: torch.Tensor,
        latent_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, T, H, W = x.shape
        y_prior = self.prior(x[:, 0], origin_month)
        ang = origin_month.float() * (2.0 * math.pi / 12.0)
        season_vec = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)

        st = self.static_map[:, :, None, :, :].expand(B, -1, T, -1, -1)
        xin = _pad_spherical_3d(torch.cat([x, st], dim=1), (0, 0), self.embed_pad_h, self.embed_pad_w)
        tok = self.patch_norm(self.patch_proj(xin).permute(0, 2, 3, 4, 1))
        tok = tok + self.season_embed1(season_vec).view(B, 1, 1, 1, -1)

        tok = self._run_blocks(self.layer1, tok)
        skip = tok
        tok = self.downsample(tok) + self.season_embed2(season_vec).view(B, 1, 1, 1, -1)
        tok = self._run_blocks(self.layer2, tok)
        tok = self.global_bottleneck(tok, self.static_coarse, season_vec, latent_noise=latent_noise)
        tok = self._run_blocks(self.layer3, tok)
        tok = self._run_blocks(self.layer4, self.upsample(tok))

        feat = self.time_to_leads(torch.cat([tok, skip], dim=-1).permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        leads_ar = torch.arange(1, self.horizon + 1, device=x.device).view(1, self.horizon)
        tgt_m = (origin_month.view(B, 1) + leads_ar).float() * (2.0 * math.pi / 12.0)
        tgt_vec = torch.stack([torch.sin(tgt_m), torch.cos(tgt_m)], dim=-1)
        feat = self.rec_norm(feat + self.lead_season_proj(tgt_vec).view(B, self.horizon, 1, 1, -1))

        feat_2d = feat.permute(0, 1, 4, 2, 3).reshape(B, self.horizon * 2 * self.embed_dim, 181, 360)
        rec = F.pixel_shuffle(self.rec_head(feat_2d), upscale_factor=4)
        top = self.embed_pad_h[0]
        rec = rec[:, :, top:top + H, :]
        trans_delta, trans_feat = rec[:, :self.horizon], rec[:, self.horizon:]

        ref_in = torch.cat([
            trans_delta, trans_feat, y_prior, x[:, 0, -3:, :, :],
            x[:, [1, 2, 3, 4, 7], -1, :, :], self.static_map[:, 4:5].expand(B, -1, -1, -1),
        ], dim=1)
        return (y_prior + trans_delta + self.refiner(ref_in)) * self.ocean_mask


@dataclass
class CanglongV2Config:
    img_size: Tuple[int, int] = (721, 1440)
    n_months: int = 16
    in_chans: int = 8
    horizon: int = 6
    default_out_leads: int = 3
    embed_dim: int = 64
    depths: Tuple[int, int, int, int] = (2, 4, 4, 2)
    num_heads: Tuple[int, int, int, int] = (4, 8, 8, 4)
    window_size: Tuple[int, int, int] = (2, 6, 12)
    use_cslim: bool = True
    cslim_version: int = 2
    n_ocean1: int = 43216
    n_eof_sst: int = 100
    n_eof_atm: int = 24
    n_core_sst: int = 16
    n_core_atm: int = 6
    n_cyc_sst: int = 16
    n_cyc_atm: int = 8
    cslim_core_dim: int = 149
    cslim_feat_dim: int = 1017
    cslim_enso_feat_dim: int = 1093
    cslim_enso_core_dim: int = 253
    n_eof_enso: int = 16
    has_ensemble: bool = False
    ens_err_rank: int = 32
    ens_ic_modes_sst: int = 24
    ens_ic_modes_atm: int = 8
    ens_ar1_rho: float = 0.82
    ens_flow_gamma: float = 0.0


class CanglongV2(nn.Module):
    """Self-contained single-checkpoint CAS-Canglong 0.25° V2/V2.5 + CS-LIM Hybrid & AI Ensemble.

    Accepts the EXACT official WeatherAI input tensor `(B, 8, 16, 721, 1440)` normalized
    by `normalized_data.pt` statistics and outputs `(B, 1, out_leads, 721, 1440)` in the
    exact same normalized SST space (`sst_K = out * 11.627 + 286.79`).

    For V2.5 Ensemble checkpoints (`cfg.has_ensemble=True`), `forecast_ensemble(...)`
    generates calibrated flow-dependent AI ensemble forecasts using Orthogonal Bred
    Vectors (OBV) + Stochastic Latent Teleconnection Perturbations + Lead/Season-Calibrated
    Coupled-EOF Stochastic Innovations.
    """

    def __init__(self, cfg: Optional[CanglongV2Config] = None):
        super().__init__()
        cfg = cfg or CanglongV2Config()
        self.cfg = cfg
        H, W = cfg.img_size

        self.backbone = Canglong025Backbone(
            img_hw=cfg.img_size,
            in_channels=cfg.in_chans,
            history=cfg.n_months,
            horizon=cfg.horizon,
            embed_dim=cfg.embed_dim,
            depths=cfg.depths,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            use_checkpoint=False,
        )

        # Official normalized_data.pt <-> training anomaly space conversion buffers
        self.register_buffer("raw_mean", torch.zeros(1, cfg.in_chans, 1, 1, 1))
        self.register_buffer("raw_std", torch.ones(1, cfg.in_chans, 1, 1, 1))
        self.register_buffer("anom_std", torch.ones(1, cfg.in_chans, 1, 1, 1))
        self.register_buffer("clim_sst_norm", torch.zeros(12, H, W, dtype=torch.float32))
        self.register_buffer("clim_atm_norm", torch.zeros(12, cfg.in_chans - 1, H, W, dtype=torch.float16))
        # Coarse (45x90) SST climatology for instant 100%-exact calendar month detection
        self.register_buffer("clim_sst_coarse", torch.zeros(12, 45, 90, dtype=torch.float32))
        self.register_buffer("ocean_coarse", torch.ones(45, 90, dtype=torch.bool))

        # Built-in Cyclostationary Coupled-EOF LIM buffers (pure PyTorch tensor ops)
        if cfg.use_cslim:
            self.register_buffer("ocean1_idx", torch.zeros(cfg.n_ocean1, dtype=torch.long))
            self.register_buffer("w_sqrt1", torch.ones(181 * 360, dtype=torch.float32))
            self.register_buffer("w_sqrt1_ocean", torch.ones(cfg.n_ocean1, dtype=torch.float32))
            self.register_buffer("inv_w1_ocean", torch.ones(cfg.n_ocean1, dtype=torch.float32))
            if cfg.cslim_version >= 2:
                self.register_buffer("w_area1_ocean", torch.full((cfg.n_ocean1,), 1.0 / max(cfg.n_ocean1, 1), dtype=torch.float32))
                self.register_buffer("G_W", torch.zeros(cfg.horizon, 3, dtype=torch.float32))
            self.register_buffer("Vt_sst", torch.zeros(cfg.n_eof_sst, cfg.n_ocean1, dtype=torch.float32))
            self.register_buffer("Vt_atm", torch.zeros(cfg.in_chans - 1, cfg.n_eof_atm, 181 * 360, dtype=torch.float16))
            self.register_buffer("f_mean", torch.zeros(cfg.cslim_feat_dim, dtype=torch.float32))
            self.register_buffer("f_std", torch.ones(cfg.cslim_feat_dim, dtype=torch.float32))
            self.register_buffer("W_cslim", torch.zeros(cfg.horizon, cfg.cslim_feat_dim, cfg.n_eof_sst, dtype=torch.float32))
            self.register_buffer("rho_res", torch.zeros(cfg.horizon, dtype=torch.float32))
            self.register_buffer("blend_w_maps", torch.zeros(1, cfg.horizon, H, 1, dtype=torch.float32))
            if cfg.cslim_version >= 4:
                self.register_buffer("f_enso_mean", torch.zeros(cfg.cslim_enso_feat_dim, dtype=torch.float32))
                self.register_buffer("f_enso_std", torch.ones(cfg.cslim_enso_feat_dim, dtype=torch.float32))
                self.register_buffer("W_cslim_enso", torch.zeros(12, cfg.horizon, cfg.cslim_enso_feat_dim, cfg.n_eof_enso, dtype=torch.float32))
                self.register_buffer("enso_waveguide_mask", torch.zeros(1, 1, H, W, dtype=torch.float32))
                self.register_buffer("blend_w_enso", torch.zeros(1, cfg.horizon, 1, 1, dtype=torch.float32))

        # Built-in Flow-Dependent AI Ensemble Perturbation buffers (strictly calibrated on Train/Val)
        if cfg.has_ensemble:
            self.register_buffer("ens_sst_pc_std", torch.ones(cfg.n_eof_sst, dtype=torch.float32))
            self.register_buffer("ens_atm_pc_std", torch.ones(cfg.in_chans - 1, cfg.n_eof_atm, dtype=torch.float32))
            self.register_buffer("ens_err_basis", torch.zeros(cfg.horizon, cfg.ens_err_rank, cfg.n_eof_sst, dtype=torch.float32))
            self.register_buffer("ens_err_g_std", torch.zeros(cfg.horizon, dtype=torch.float32))
            self.register_buffer("ens_season_scale", torch.ones(12, cfg.horizon, dtype=torch.float32))
            self.register_buffer("ens_ic_scale", torch.tensor(0.20, dtype=torch.float32))
            self.register_buffer("ens_latent_scale", torch.tensor(0.18, dtype=torch.float32))
            self.register_buffer("ens_stoch_scale", torch.ones(cfg.horizon, dtype=torch.float32))

    @classmethod
    def from_checkpoint_dict(cls, obj: dict) -> "CanglongV2":
        raw_cfg = dict(obj["config"])
        for k in ("img_size", "depths", "num_heads", "window_size"):
            if k in raw_cfg:
                raw_cfg[k] = tuple(raw_cfg[k])
        # Backward compatibility with older V2.1 checkpoints missing newer config keys
        valid_fields = set(CanglongV2Config.__dataclass_fields__.keys())
        filtered = {k: v for k, v in raw_cfg.items() if k in valid_fields}
        cfg = CanglongV2Config(**filtered)
        model = cls(cfg)
        model.load_state_dict(obj["state_dict"], strict=True)
        return model

    def to_checkpoint_dict(self) -> dict:
        return {
            "format": "canglong_v25_hybrid" if (self.cfg.cslim_version >= 3 or self.cfg.has_ensemble) else "canglong_v2_hybrid",
            "config": asdict(self.cfg),
            "state_dict": self.state_dict(),
        }

    def detect_origin_month(self, x_raw_norm: torch.Tensor) -> torch.Tensor:
        """Infer exact calendar month (0..11) of the last input frame x[:, :, -1] from the
        16-month seasonal cycle of global ocean SST (100% exact on 1959–2026)."""
        B, _, T, _, _ = x_raw_norm.shape
        sst_c = F.adaptive_avg_pool2d(x_raw_norm[:, 0], (45, 90))  # (B, T, 45, 90)
        mask = self.ocean_coarse  # (45, 90)
        obs_oc = sst_c[:, :, mask]  # (B, T, N_oc)
        clim_oc = self.clim_sst_coarse[:, mask]  # (12, N_oc)
        offsets = torch.arange(T, device=x_raw_norm.device) - (T - 1)  # [-15, ..., 0]
        cand_m = (torch.arange(12, device=x_raw_norm.device)[:, None] + offsets[None, :]) % 12  # (12, T)
        ref = clim_oc[cand_m]  # (12, T, N_oc)
        # MSE per candidate origin month: (B, 12)
        err = (obs_oc.unsqueeze(1) - ref.unsqueeze(0)).square().mean(dim=(2, 3))
        return err.argmin(dim=1).long()

    def _raw_to_anom(self, x_raw_norm: torch.Tensor, origin_month: torch.Tensor) -> torch.Tensor:
        B, _, T, H, W = x_raw_norm.shape
        offsets = torch.arange(T, device=x_raw_norm.device) - (T - 1)
        in_m = (origin_month[:, None] + offsets[None, :]) % 12  # (B, 16)
        scale = self.raw_std / self.anom_std  # (1, 8, 1, 1, 1)

        c_sst = self.clim_sst_norm[in_m]  # (B, 16, H, W)
        x_sst = ((x_raw_norm[:, 0] - c_sst) * scale[:, 0, :, :, :]) * self.backbone.ocean_mask
        # Process atmospheric channels cleanly without materializing double-sized fp32 intermediates
        x_atm = torch.empty((B, self.cfg.in_chans - 1, T, H, W), dtype=x_raw_norm.dtype, device=x_raw_norm.device)
        for ci in range(self.cfg.in_chans - 1):
            c_ci = self.clim_atm_norm[in_m, ci].to(dtype=x_raw_norm.dtype)  # (B, 16, H, W)
            x_atm[:, ci] = (x_raw_norm[:, ci + 1] - c_ci) * scale[:, ci + 1, :, :, :]
        return torch.cat([x_sst.unsqueeze(1), x_atm], dim=1)

    def _forward_cslim(self, x_anom: torch.Tensor, origin_month: torch.Tensor, out_leads: int) -> torch.Tensor:
        """Pure-PyTorch Warming-Safe Mode-Adaptive Cyclostationary Coupled-EOF LIM branch."""
        B, C, T, H, W = x_anom.shape
        sst_1 = F.interpolate(x_anom[:, 0], size=(181, 360), mode="area").reshape(B, T, 181 * 360)
        sst_1_oc = sst_1[:, :, self.ocean1_idx]  # (B, 16, n_oc1)

        if self.cfg.cslim_version >= 2:
            # Separate global ocean mean SSTA first so warming never gets shrunk by Ridge!
            g_mean = (sst_1_oc * self.w_area1_ocean.view(1, 1, -1)).sum(dim=-1)  # (B, 16)
            g_bg12 = g_mean[:, -12:].mean(dim=1)  # (B,)
            g_fast = g_mean[:, -1] - g_bg12
            g_tend = g_mean[:, -1] - g_mean[:, -2]
            g_feat = torch.stack([g_fast, g_tend, torch.ones_like(g_fast)], dim=-1)  # (B, 3)
            g_pred = g_bg12.unsqueeze(1) + torch.matmul(g_feat, self.G_W[:out_leads].t())  # (B, L)
            g_orig = g_mean[:, -1]  # (B,)
            sst_oc = (sst_1_oc - g_mean.unsqueeze(-1)) * self.w_sqrt1_ocean.view(1, 1, -1)
        else:
            g_pred = None
            g_orig = None
            sst_oc = sst_1_oc * self.w_sqrt1_ocean.view(1, 1, -1)

        pc_sst = torch.matmul(sst_oc, self.Vt_sst.t())  # (B, 16, K_sst)

        p0, p1, p2 = pc_sst[:, -1], pc_sst[:, -2], pc_sst[:, -3]
        p_bg6 = pc_sst[:, -6:].mean(dim=1)
        p_bg15 = pc_sst.mean(dim=1)

        ang = origin_month.float() * (2.0 * math.pi / 12.0)
        s1 = torch.sin(ang).unsqueeze(1)
        c1 = torch.cos(ang).unsqueeze(1)
        s2 = torch.sin(2.0 * ang).unsqueeze(1)
        c2 = torch.cos(2.0 * ang).unsqueeze(1)
        ones = torch.ones_like(s1)

        if self.cfg.cslim_version == 4:
            # V4: Dual-Branch Pareto Operator
            #   Branch 1 (Global & All 8 Paper Regions): Exact V2.1-clean feature vector -> W_cslim (16, F, 100)
            #   Branch 2 (Tropical Pacific ENSO Waveguide): Month-Conditioned Cyclostationary Phase-Locked
            #             Non-Mean-Reverting Operator -> W_cslim_enso (12, 16, F_enso, 16)
            atm_last6 = x_anom[:, 1:, -6:, :, :].reshape(B * (C - 1), 6, H, W)
            atm_1 = F.interpolate(atm_last6, size=(181, 360), mode="area").reshape(B, C - 1, 6, 181 * 360)
            atm_1 = atm_1 * self.w_sqrt1.view(1, 1, 1, -1)
            nc_a = self.cfg.n_core_atm
            pc_a0_list, pc_a3_list, pc_acyc_list = [], [], []
            a0_c_list, a6_c_list = [], []
            for ci in range(C - 1):
                p_c = torch.matmul(atm_1[:, ci], self.Vt_atm[ci].float().t())  # (B, 6, K_atm)
                pc_a0_list.append(p_c[:, -1])
                pc_a3_list.append(p_c[:, -3:].mean(dim=1))
                pc_acyc_list.append(p_c[:, -1, :self.cfg.n_cyc_atm])
                a0_c_list.append(p_c[:, -1, :nc_a])
                a6_c_list.append(p_c[:, :, :nc_a].mean(dim=1))
            a0 = torch.cat(pc_a0_list, dim=-1)
            a3 = torch.cat(pc_a3_list, dim=-1)
            a_top = torch.cat(pc_acyc_list, dim=-1)
            a0_c = torch.cat(a0_c_list, dim=-1)
            a6_c = torch.cat(a6_c_list, dim=-1)

            nc = self.cfg.n_core_sst
            p0c, p1c, p2c = p0[:, :nc], p1[:, :nc], p2[:, :nc]
            tail = torch.cat([
                p0[:, nc:], p1[:, nc:], p2[:, nc:],
                p_bg6[:, nc:], p_bg15[:, nc:],
                a0, a3, a_top * s1, a_top * c1,
            ], dim=-1)

            # 1) Global 8-Region branch (exact V2.1 core + tail)
            core_g = torch.cat([
                p0c, p1c, p2c, p0c - p2c, p_bg15[:, :nc],
                p0c * s1, p0c * c1, p0c * s2, p0c * c2,
                s1, c1, s2, c2, ones,
            ], dim=-1)
            feat_g = torch.cat([core_g, tail], dim=-1)
            feat_n = (feat_g - self.f_mean.unsqueeze(0)) / self.f_std.unsqueeze(0)
            feat_n[:, self.cfg.cslim_core_dim - 1] = 1.0

            # 2) Month-Conditioned Cyclostationary Phase-Locked ENSO branch
            p3, p5 = pc_sst[:, -4], pc_sst[:, -6]
            d1 = p0c - p1c
            d2 = p0c - p2c
            d3 = p0c - p3[:, :nc]
            d5 = p0c - p5[:, :nc]
            core_e = torch.cat([
                p0c, p1c, p2c, d1, d2, d3, d5, p_bg15[:, :nc],
                p0c * s1, p0c * c1, p0c * s2, p0c * c2,
                a0_c, a6_c,
                s1, c1, s2, c2, ones,
            ], dim=-1)
            feat_e = torch.cat([core_e, tail], dim=-1)
            feat_en = (feat_e - self.f_enso_mean.unsqueeze(0)) / self.f_enso_std.unsqueeze(0)
            feat_en[:, self.cfg.cslim_enso_core_dim - 1] = 1.0
            W_e_b = self.W_cslim_enso[origin_month, :out_leads]  # (B, L, F_enso, K_enso)
            pc_enso_pred = torch.einsum("bf,blfk->blk", feat_en, W_e_b)
        elif self.cfg.cslim_version == 3:
            # V3: Full 16-month atmospheric recharge integrals (WWV proxy) + coupled core + ENSO skewness
            atm_all = x_anom[:, 1:, :, :, :].reshape(B * (C - 1), T, H, W)
            atm_1 = F.interpolate(atm_all, size=(181, 360), mode="area").reshape(B, C - 1, T, 181 * 360)
            atm_1 = atm_1 * self.w_sqrt1.view(1, 1, 1, -1)
            nc_a = self.cfg.n_core_atm
            k_a6c = min(3, nc_a)
            a0_c_l, a6_c_l, a12_c_l, a6_cyc_l = [], [], [], []
            a0_t_l, a3_t_l, a6_t_l, a_cyc_l = [], [], [], []
            for ci in range(C - 1):
                p_c = torch.matmul(atm_1[:, ci], self.Vt_atm[ci].float().t())  # (B, 16, K_atm)
                a0_c_l.append(p_c[:, -1, :nc_a])
                a6_c_l.append(p_c[:, -6:, :nc_a].mean(dim=1))
                a12_c_l.append(p_c[:, -12:, :nc_a].mean(dim=1))
                a6_cyc_l.append(p_c[:, -6:, :k_a6c].mean(dim=1))
                a0_t_l.append(p_c[:, -1, nc_a:])
                a3_t_l.append(p_c[:, -3:, nc_a:].mean(dim=1))
                a6_t_l.append(p_c[:, -6:, nc_a:].mean(dim=1))
                a_cyc_l.append(p_c[:, -1, :self.cfg.n_cyc_atm])
            a0_core = torch.cat(a0_c_l, dim=-1)
            a6_core = torch.cat(a6_c_l, dim=-1)
            a12_core = torch.cat(a12_c_l, dim=-1)
            a6_cyc = torch.cat(a6_cyc_l, dim=-1)
            a0_tail = torch.cat(a0_t_l, dim=-1)
            a3_tail = torch.cat(a3_t_l, dim=-1)
            a6_tail = torch.cat(a6_t_l, dim=-1)
            a_top = torch.cat(a_cyc_l, dim=-1)

            nc = self.cfg.n_core_sst
            ns = self.cfg.n_cyc_sst
            p5 = pc_sst[:, -6]
            p0c, p1c, p2c, p5c = p0[:, :nc], p1[:, :nc], p2[:, :nc], p5[:, :nc]
            p0s = p0[:, :ns]
            q1 = p0[:, :4].square()
            q2 = torch.stack([p0[:, 0] * p0[:, 1], p0[:, 0] * p0[:, 2], p0[:, 1] * p0[:, 2]], dim=-1)
            q3 = p0[:, :3] * (p0[:, :3] - p2[:, :3])
            q4 = torch.stack([p0[:, 0] * a6_core[:, 0], p0[:, 0] * a12_core[:, 0]], dim=-1)
            core = torch.cat([
                p0c, p1c, p2c, p0c - p2c, p0c - p5c,
                p_bg6[:, :nc], p_bg15[:, :nc],
                p0s * s1, p0s * c1, p0s * s2, p0s * c2,
                a0_core, a6_core, a12_core,
                a6_cyc * s1, a6_cyc * c1,
                q1, q2, q3, q4,
                s1, c1, s2, c2, ones,
            ], dim=-1)
            tail = torch.cat([
                p0[:, nc:], p1[:, nc:], p2[:, nc:],
                p_bg6[:, nc:], p_bg15[:, nc:],
                a0_tail, a3_tail, a6_tail,
                a_top * s1, a_top * c1,
            ], dim=-1)
            feat = torch.cat([core, tail], dim=-1)
            feat_n = (feat - self.f_mean.unsqueeze(0)) / self.f_std.unsqueeze(0)
            feat_n[:, self.cfg.cslim_core_dim - 1] = 1.0
            pc_enso_pred = None
        else:
            # V1/V2: Last 3 frames of atmospheric channels 1..7 -> (B, 7, 3, 181*360)
            atm_last3 = x_anom[:, 1:, -3:, :, :].reshape(B * (C - 1), 3, H, W)
            atm_1 = F.interpolate(atm_last3, size=(181, 360), mode="area").reshape(B, C - 1, 3, 181 * 360)
            atm_1 = atm_1 * self.w_sqrt1.view(1, 1, 1, -1)
            pc_a0_list, pc_a3_list, pc_acyc_list = [], [], []
            for ci in range(C - 1):
                p_c = torch.matmul(atm_1[:, ci], self.Vt_atm[ci].float().t())  # (B, 3, K_atm)
                pc_a0_list.append(p_c[:, -1])
                pc_a3_list.append(p_c.mean(dim=1))
                pc_acyc_list.append(p_c[:, -1, :self.cfg.n_cyc_atm])
            a0 = torch.cat(pc_a0_list, dim=-1)
            a3 = torch.cat(pc_a3_list, dim=-1)
            a_top = torch.cat(pc_acyc_list, dim=-1)

            if self.cfg.cslim_version >= 2:
                nc = self.cfg.n_core_sst
                p0c, p1c, p2c = p0[:, :nc], p1[:, :nc], p2[:, :nc]
                core = torch.cat([
                    p0c, p1c, p2c, p0c - p2c, p_bg15[:, :nc],
                    p0c * s1, p0c * c1, p0c * s2, p0c * c2,
                    s1, c1, s2, c2, ones,
                ], dim=-1)
                tail = torch.cat([
                    p0[:, nc:], p1[:, nc:], p2[:, nc:],
                    p_bg6[:, nc:], p_bg15[:, nc:],
                    a0, a3, a_top * s1, a_top * c1,
                ], dim=-1)
                feat = torch.cat([core, tail], dim=-1)
                feat_n = (feat - self.f_mean.unsqueeze(0)) / self.f_std.unsqueeze(0)
                feat_n[:, self.cfg.cslim_core_dim - 1] = 1.0
                pc_enso_pred = None
            else:
                p_tend = (p0 - p2)[:, :30]
                p_top = p0[:, :self.cfg.n_cyc_sst]
                feat = torch.cat([
                    p0, p1, p2, p_tend, p_bg6, p_bg15, a0, a3,
                    p_top * s1, p_top * c1, p_top * s2, p_top * c2,
                    a_top * s1, a_top * c1,
                    s1, c1, s2, c2, ones,
                ], dim=-1)
                feat_n = (feat - self.f_mean.unsqueeze(0)) / self.f_std.unsqueeze(0)
                feat_n[:, -1] = 1.0
                pc_enso_pred = None

        pc_pred = torch.einsum("bf,lfk->blk", feat_n, self.W_cslim[:out_leads])

        def _reconstruct_025(pcs: torch.Tensor, g_add: Optional[torch.Tensor]) -> torch.Tensor:
            b_sz, m_sz, _ = pcs.shape
            rec_oc = torch.matmul(pcs.float(), self.Vt_sst.float()) * self.inv_w1_ocean.float().view(1, 1, -1)
            grid1 = torch.zeros((b_sz, m_sz, 181 * 360), dtype=rec_oc.dtype, device=pcs.device)
            grid1.scatter_(2, self.ocean1_idx.view(1, 1, -1).expand(b_sz, m_sz, -1), rec_oc)
            up = F.interpolate(grid1.view(b_sz, m_sz, 181, 360), size=(H, W), mode="bilinear", align_corners=False)
            if g_add is not None:
                up = up + g_add.float().view(b_sz, m_sz, 1, 1)
            return up * self.backbone.ocean_mask

        origin_coarse_025 = _reconstruct_025(p0.unsqueeze(1), None if g_orig is None else g_orig.unsqueeze(1))[:, 0]
        origin_fine = (x_anom[:, 0, -1] - origin_coarse_025) * self.backbone.ocean_mask.squeeze(1)
        pred_coarse = _reconstruct_025(pc_pred, g_pred)
        if pc_enso_pred is not None:
            pc_enso_full = pc_pred.clone()
            pc_enso_full[:, :, :self.cfg.n_eof_enso] = pc_enso_pred
            pred_enso_coarse = _reconstruct_025(pc_enso_full, g_pred)
            mw = self.enso_waveguide_mask
            pred_coarse = (1.0 - mw) * pred_coarse + mw * pred_enso_coarse
        pred_025 = pred_coarse + self.rho_res[:out_leads].view(1, out_leads, 1, 1) * origin_fine.unsqueeze(1)
        return pred_025 * self.backbone.ocean_mask

    def _ssta_to_raw_norm(self, pred_ssta: torch.Tensor, origin_month: torch.Tensor) -> torch.Tensor:
        L = pred_ssta.shape[1]
        leads_ar = torch.arange(1, L + 1, device=pred_ssta.device).view(1, L)
        tgt_m = (origin_month[:, None] + leads_ar) % 12
        clim_tgt = self.clim_sst_norm[tgt_m]
        ssta_to_raw = self.anom_std[0, 0, 0, 0, 0] / self.raw_std[0, 0, 0, 0, 0]
        return (pred_ssta * ssta_to_raw + clim_tgt) * self.backbone.ocean_mask

    def _forward_hybrid_anom(
        self,
        x_anom: torch.Tensor,
        origin_month: torch.Tensor,
        out_leads: int,
        latent_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_nn = self.backbone(x_anom, origin_month, latent_noise=latent_noise)[:, :out_leads].float()
        if self.cfg.use_cslim:
            with torch.amp.autocast(device_type=x_anom.device.type, enabled=False):
                pred_lim = self._forward_cslim(x_anom.float(), origin_month, out_leads)
            wm = self.blend_w_maps[:, :out_leads]
            if self.cfg.cslim_version >= 4:
                mw = self.enso_waveguide_mask
                we = self.blend_w_enso[:, :out_leads]
                wm = (1.0 - mw) * wm + mw * torch.maximum(wm, we)
            return (1.0 - wm) * pred_nn + wm * pred_lim
        return pred_nn

    def _reconstruct_sst_eof_025(self, pcs: torch.Tensor, g_add: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Map SST EOF coefficients `(B, M, K_sst)` to native 0.25° ocean-masked fields `(B, M, H, W)`."""
        H, W = self.cfg.img_size
        b_sz, m_sz, k_in = pcs.shape
        rec_oc = torch.matmul(pcs.float(), self.Vt_sst[:k_in].float()) * self.inv_w1_ocean.float().view(1, 1, -1)
        grid1 = torch.zeros((b_sz, m_sz, 181 * 360), dtype=rec_oc.dtype, device=pcs.device)
        grid1.scatter_(2, self.ocean1_idx.view(1, 1, -1).expand(b_sz, m_sz, -1), rec_oc)
        up = F.interpolate(grid1.view(b_sz, m_sz, 181, 360), size=(H, W), mode="bilinear", align_corners=False)
        if g_add is not None:
            up = up + g_add.float().view(b_sz, m_sz, 1, 1)
        return up * self.backbone.ocean_mask

    def _reconstruct_atm_eof_025(self, atm_pcs: torch.Tensor) -> torch.Tensor:
        """Map coupled atmospheric EOF coefficients `(M, C-1, K_atm)` -> `(M, C-1, H, W)`."""
        H, W = self.cfg.img_size
        M, n_atm, k_in = atm_pcs.shape
        inv_w_all = (1.0 / self.w_sqrt1).view(1, -1)
        outs = []
        for ci in range(n_atm):
            g1 = (torch.matmul(atm_pcs[:, ci], self.Vt_atm[ci, :k_in].float()) * inv_w_all).view(M, 1, 181, 360)
            outs.append(F.interpolate(g1, size=(H, W), mode="bilinear", align_corners=False))
        return torch.cat(outs, dim=1)  # (M, C-1, H, W)

    @staticmethod
    def _gram_schmidt_rows(vecs: torch.Tensor) -> torch.Tensor:
        """Orthonormalize rows of `vecs: (M, D)` with safe fallback if M > D."""
        M, D = vecs.shape
        out = []
        for i in range(M):
            v = vecs[i].clone()
            for u in out:
                v = v - torch.dot(v, u) * u
            nrm = v.norm()
            if nrm > 1e-6:
                v = v / nrm
            else:
                v = vecs[i] / (vecs[i].norm() + 1e-6)
            out.append(v)
        return torch.stack(out, dim=0)

    @torch.no_grad()
    def forecast_ensemble(
        self,
        x: torch.Tensor,
        n_members: int = 32,
        origin_month: Optional[torch.Tensor] = None,
        out_leads: Optional[int] = None,
        method: str = "hybrid_obv",
        n_breed_cycles: int = 2,
        seed: Optional[int] = 42,
        batch_size: int = 4,
        output_device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        """Generate flow-dependent AI-perturbed ensemble forecasts up to 16 months.

        Combines three modern AI ensemble perturbation mechanisms:
          1. Flow-dependent Orthogonal AI Bred Vectors (OBV) in coupled ocean-atmosphere
             EOF space with antithetic pairing (+δ, -δ) for unbiased ensemble mean.
          2. Stochastic latent teleconnection bottleneck perturbations representing
             unresolved subgrid atmospheric forcing (WWB / MJO).
          3. Lead- & season-calibrated AR(1) temporally coherent coupled-EOF stochastic
             innovations matched to validation CRPS & Spread-Skill ≈ 1.0.

        Returns `(B, n_members, 1, L, H, W)` normalized SST (`* 11.627 + 286.79` -> K).
        """
        assert self.cfg.use_cslim, "forecast_ensemble requires CS-LIM EOF basis (`use_cslim=True`)."
        assert method in ("hybrid_obv", "obv", "latent"), f"Unknown ensemble method: {method}"
        L = self.cfg.default_out_leads if out_leads is None else int(out_leads)
        assert 1 <= L <= self.cfg.horizon, f"out_leads must be in 1..{self.cfg.horizon}, got {L}"
        assert n_members >= 2, "n_members must be >= 2"

        x_in = torch.nan_to_num(x.float(), nan=0.0)
        B, C, T, H, W = x_in.shape
        dev = x_in.device
        if origin_month is None:
            origin_month = self.detect_origin_month(x_in)
        else:
            origin_month = origin_month.to(device=dev, dtype=torch.long)

        x_anom_all = self._raw_to_anom(x_in, origin_month)  # (B, 8, 16, H, W)
        gen = torch.Generator(device=dev)
        if seed is not None:
            gen.manual_seed(int(seed))

        # Calibrated scales (fallback defaults if checkpoint was exported without calibration buffers)
        ic_scale = float(self.ens_ic_scale.item()) if self.cfg.has_ensemble else 0.18
        lat_scale = float(self.ens_latent_scale.item()) if self.cfg.has_ensemble else 0.16
        stoch_scale = self.ens_stoch_scale[:L] if self.cfg.has_ensemble else torch.zeros(L, device=dev)
        sst_pc_std = self.ens_sst_pc_std if self.cfg.has_ensemble else torch.ones(self.cfg.n_eof_sst, device=dev)
        atm_pc_std = self.ens_atm_pc_std if self.cfg.has_ensemble else torch.ones(C - 1, self.cfg.n_eof_atm, device=dev)

        k_sst = min(self.cfg.ens_ic_modes_sst, self.cfg.n_eof_sst)
        k_atm = min(self.cfg.ens_ic_modes_atm, self.cfg.n_eof_atm)
        m_pairs = n_members // 2
        has_ctrl = (n_members % 2 == 1)

        # Smooth ramp weights over trailing 3 months so perturbations have zero shock at t0
        t_ramp = torch.tensor([0.35, 0.70, 1.00], device=dev, dtype=torch.float32).view(1, 1, 3, 1, 1)
        ch_b, cw_b = self.backbone.global_bottleneck.coarse_hw
        D_lat = T // 2
        C_lat = self.cfg.embed_dim * 2
        rho_t = float(self.cfg.ens_ar1_rho)
        innov_w = math.sqrt(max(1.0 - rho_t * rho_t, 1e-4))

        out_dev = dev if output_device is None else torch.device(output_device)
        ens_out = torch.empty((B, n_members, 1, L, H, W), dtype=torch.float32, device=out_dev)

        for bi in range(B):
            xa_b = x_anom_all[bi:bi + 1]   # (1, C, T, H, W)
            om_b = origin_month[bi:bi + 1]  # (1,)
            m_idx = int(om_b.item())

            # Flow-dependent spread amplifier for active developing/peaking ENSO regimes
            flow_mult = 1.0
            if float(self.cfg.ens_flow_gamma) > 0.0:
                sst_last3 = F.interpolate(xa_b[:, 0, -3:], size=(181, 360), mode="area").reshape(1, 3, 181 * 360)
                oc_last3 = sst_last3[:, :, self.ocean1_idx]
                if self.cfg.cslim_version >= 2:
                    oc_last3 = oc_last3 - (oc_last3 * self.w_area1_ocean.view(1, 1, -1)).sum(dim=-1, keepdim=True)
                pc_last3 = torch.matmul(oc_last3 * self.w_sqrt1_ocean.view(1, 1, -1), self.Vt_sst[:8].t())  # (1, 3, 8)
                std8 = sst_pc_std[:8].clamp(min=1e-4)
                e_state = float(((pc_last3[0, -1] / std8).square().mean()).sqrt().item())
                e_vel = float((((pc_last3[0, -1] - pc_last3[0, 0]) / std8).square().mean()).sqrt().item())
                e_flow = 0.65 * e_state + 0.85 * e_vel
                flow_mult = 1.0 + float(self.cfg.ens_flow_gamma) * math.tanh(max(0.0, e_flow - 0.55))
            ic_scale_b = ic_scale * (1.0 + 0.5 * (flow_mult - 1.0))

            # --- Step 1: Orthogonal AI Bred Vectors (OBV) in coupled ocean-atmosphere EOF space ---
            if method in ("hybrid_obv", "obv") and ic_scale_b > 0.0 and m_pairs > 0:
                d_sst = torch.randn((m_pairs, k_sst), generator=gen, device=dev) * sst_pc_std[:k_sst].unsqueeze(0)
                d_atm = torch.randn((m_pairs, C - 1, k_atm), generator=gen, device=dev) * atm_pc_std[:, :k_atm].unsqueeze(0)
                # Orthogonalize & normalize to calibrated initial-condition norm
                joint = torch.cat([d_sst, d_atm.reshape(m_pairs, -1)], dim=1)
                joint_o = self._gram_schmidt_rows(joint)
                target_norm_sst = float(sst_pc_std[:k_sst].norm().item()) * ic_scale_b
                target_norm_atm = float(atm_pc_std[:, :k_atm].norm().item()) * ic_scale_b
                d_sst = joint_o[:, :k_sst]
                d_sst = d_sst / (d_sst.norm(dim=1, keepdim=True) + 1e-6) * target_norm_sst
                d_atm = joint_o[:, k_sst:].reshape(m_pairs, C - 1, k_atm)
                d_atm = d_atm / (d_atm.reshape(m_pairs, -1).norm(dim=1, keepdim=True).unsqueeze(-1) + 1e-6) * target_norm_atm

                # Nonlinear AI breeding cycles through our exact hybrid operator
                breed_leads = min(2, L)
                for _ in range(max(0, int(n_breed_cycles))):
                    grown_pcs = []
                    for s0 in range(0, m_pairs, batch_size):
                        s1 = min(m_pairs, s0 + batch_size)
                        bs = s1 - s0
                        ds_b = self._reconstruct_sst_eof_025(d_sst[s0:s1].unsqueeze(0))[0]
                        da_b = self._reconstruct_atm_eof_025(d_atm[s0:s1])
                        delta_b = torch.cat([ds_b.unsqueeze(1), da_b], dim=1)
                        xa_rep = xa_b.expand(bs, -1, -1, -1, -1).clone()
                        om_rep = om_b.expand(bs)
                        d_win = delta_b.unsqueeze(2) * t_ramp  # (bs, C, 3, H, W)
                        xp = xa_rep.clone(); xp[:, :, -3:] = xp[:, :, -3:] + d_win
                        xm = xa_rep.clone(); xm[:, :, -3:] = xm[:, :, -3:] - d_win
                        pp = self._forward_hybrid_anom(xp, om_rep, breed_leads)
                        pm = self._forward_hybrid_anom(xm, om_rep, breed_leads)
                        # Growing nonlinear response (mean of L1..L2 difference) projected onto coupled SST EOFs
                        diff_sst = 0.5 * (pp - pm).mean(dim=1, keepdim=True)  # (bs, 1, H, W)
                        d1 = F.interpolate(diff_sst, size=(181, 360), mode="area").reshape(bs, 181 * 360)
                        d1_oc = d1[:, self.ocean1_idx]
                        if self.cfg.cslim_version >= 2:
                            d1_oc = d1_oc - (d1_oc * self.w_area1_ocean.unsqueeze(0)).sum(dim=-1, keepdim=True)
                        d1_w = d1_oc * self.w_sqrt1_ocean.unsqueeze(0)
                        grown_pcs.append(torch.matmul(d1_w, self.Vt_sst[:k_sst].t()))
                    g_sst = torch.cat(grown_pcs, dim=0)  # (m_pairs, k_sst)
                    g_sst = self._gram_schmidt_rows(g_sst) * target_norm_sst
                    d_sst = g_sst
                ic_pcs = (d_sst, d_atm)
            else:
                ic_pcs = None

            # --- Step 2: Stochastic Latent Teleconnection Perturbations (Antithetic Pairs) ---
            if method in ("hybrid_obv", "latent") and lat_scale > 0.0 and m_pairs > 0:
                lat_eps = torch.randn((m_pairs, D_lat, ch_b * cw_b, C_lat), generator=gen, device=dev) * lat_scale
            else:
                lat_eps = None

            # --- Step 3: Temporally Coherent AR(1) Lead- & Season-Calibrated Coupled-EOF Innovations ---
            if method in ("hybrid_obv", "latent") and self.cfg.has_ensemble and m_pairs > 0 and float(stoch_scale.max().item()) > 0.0:
                r_err = self.cfg.ens_err_rank
                # Generate AR(1) correlated Gaussian latent sequences across leads 0..L-1 for smooth trajectories
                z_seq = torch.empty((m_pairs, L, r_err), device=dev, dtype=torch.float32)
                zg_seq = torch.empty((m_pairs, L), device=dev, dtype=torch.float32)
                z_cur = torch.randn((m_pairs, r_err), generator=gen, device=dev)
                zg_cur = torch.randn((m_pairs,), generator=gen, device=dev)
                z_seq[:, 0] = z_cur
                zg_seq[:, 0] = zg_cur
                for ld in range(1, L):
                    z_cur = rho_t * z_cur + innov_w * torch.randn((m_pairs, r_err), generator=gen, device=dev)
                    zg_cur = rho_t * zg_cur + innov_w * torch.randn((m_pairs,), generator=gen, device=dev)
                    z_seq[:, ld] = z_cur
                    zg_seq[:, ld] = zg_cur
                # Project through lead-specific forecast-error covariance basis & seasonal multiplier
                s_mult = (self.ens_season_scale[m_idx, :L] * stoch_scale * flow_mult).view(1, L, 1)
                pc_stoch = torch.einsum("mlr,lrk->mlk", z_seq, self.ens_err_basis[:L]) * s_mult  # (m_pairs, L, K_sst)
                g_stoch = zg_seq * self.ens_err_g_std[:L].unsqueeze(0) * s_mult.squeeze(-1)     # (m_pairs, L)
            else:
                pc_stoch = None
                g_stoch = None

            # --- Step 4: Batched Antithetic Forward Pass -> Full 0.25° Members ---
            out_idx = 0
            if has_ctrl:
                p_ctrl = self._forward_hybrid_anom(xa_b, om_b, L)
                ens_out[bi, out_idx, 0].copy_(self._ssta_to_raw_norm(p_ctrl, om_b)[0].to(out_dev))
                out_idx += 1

            for s0 in range(0, m_pairs, max(1, batch_size // 2)):
                s1 = min(m_pairs, s0 + max(1, batch_size // 2))
                bs = s1 - s0
                xa_p = xa_b.expand(bs, -1, -1, -1, -1).clone()
                xa_m = xa_b.expand(bs, -1, -1, -1, -1).clone()
                if ic_pcs is not None:
                    ds_b = self._reconstruct_sst_eof_025(ic_pcs[0][s0:s1].unsqueeze(0))[0]
                    da_b = self._reconstruct_atm_eof_025(ic_pcs[1][s0:s1])
                    d_win = torch.cat([ds_b.unsqueeze(1), da_b], dim=1).unsqueeze(2) * t_ramp
                    xa_p[:, :, -3:] = xa_p[:, :, -3:] + d_win
                    xa_m[:, :, -3:] = xa_m[:, :, -3:] - d_win
                if batch_size <= 1:
                    ln_p = lat_eps[s0:s1] if lat_eps is not None else None
                    ln_m = -lat_eps[s0:s1] if lat_eps is not None else None
                    om_1 = om_b.expand(bs)
                    pred_p = self._forward_hybrid_anom(xa_p, om_1, L, latent_noise=ln_p)
                    pred_m = self._forward_hybrid_anom(xa_m, om_1, L, latent_noise=ln_m)
                    pred_cat = torch.cat([pred_p, pred_m], dim=0)
                    om_cat = om_b.expand(2 * bs)
                else:
                    xa_cat = torch.cat([xa_p, xa_m], dim=0)
                    om_cat = om_b.expand(2 * bs)
                    ln_cat = torch.cat([lat_eps[s0:s1], -lat_eps[s0:s1]], dim=0) if lat_eps is not None else None
                    pred_cat = self._forward_hybrid_anom(xa_cat, om_cat, L, latent_noise=ln_cat)  # (2*bs, L, H, W)

                if pc_stoch is not None:
                    st_fields = self._reconstruct_sst_eof_025(pc_stoch[s0:s1], g_stoch[s0:s1])  # (bs, L, H, W)
                    pred_cat[:bs] = pred_cat[:bs] + st_fields
                    pred_cat[bs:] = pred_cat[bs:] - st_fields

                raw_cat = self._ssta_to_raw_norm(pred_cat, om_cat).to(out_dev)  # (2*bs, L, H, W)
                for k in range(bs):
                    ens_out[bi, out_idx, 0].copy_(raw_cat[k])
                    ens_out[bi, out_idx + 1, 0].copy_(raw_cat[bs + k])
                    out_idx += 2

        return ens_out

    def forward(
        self,
        x: torch.Tensor,
        origin_month: Optional[torch.Tensor] = None,
        out_leads: Optional[int] = None,
        ensemble_members: Optional[int] = None,
    ) -> torch.Tensor:
        """x: (B, 8, 16, 721, 1440) official normalized ERA5 -> (B, 1, out_leads, 721, 1440) normalized SST."""
        if ensemble_members is not None and ensemble_members >= 2:
            ens = self.forecast_ensemble(x, n_members=ensemble_members, origin_month=origin_month, out_leads=out_leads)
            return ens.mean(dim=1)
        L = self.cfg.default_out_leads if out_leads is None else int(out_leads)
        assert 1 <= L <= self.cfg.horizon, f"out_leads must be in 1..{self.cfg.horizon}, got {L}"
        x_in = torch.nan_to_num(x.float(), nan=0.0)
        if origin_month is None:
            origin_month = self.detect_origin_month(x_in)
        else:
            origin_month = origin_month.to(device=x_in.device, dtype=torch.long)

        x_anom = self._raw_to_anom(x_in, origin_month)
        pred_ssta = self._forward_hybrid_anom(x_anom, origin_month, L)
        sst_raw_norm = self._ssta_to_raw_norm(pred_ssta, origin_month)
        return sst_raw_norm.unsqueeze(1)  # (B, 1, L, 721, 1440)


def load_canglong_v2(path: str, map_location="cpu") -> CanglongV2:
    """Standalone loader for CAS-Canglong 0.25° V2/V2.6 single-file .tar or .tar.gz checkpoints."""
    import gzip
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rb") as gzf:
            obj = torch.load(gzf, map_location=map_location, weights_only=True)
    else:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    return CanglongV2.from_checkpoint_dict(obj).to(map_location).eval()
