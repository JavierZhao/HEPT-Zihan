import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# --- Monkey-patch Conv2d to support FLOPs counting ---
# We record the last output shape and compute FLOPs accordingly.
from torch.nn import Conv2d

# Save original forward
_orig_conv2d_forward = Conv2d.forward


def _conv2d_forward_and_record(self, input):
    output = _orig_conv2d_forward(self, input)
    # record output shape for FLOPs computation
    self.__last_output_shape__ = output.shape
    return output


Conv2d.forward = _conv2d_forward_and_record

# Ensure Conv2d modules always have a __flops__ attribute
Conv2d.__flops__ = 0  # default


# Attach accumulate_flops to Conv2d
def _conv2d_accumulate_flops(self):
    """
    Compute and record FLOPs for this Conv2d based on last output shape.
    """
    shape = getattr(self, "__last_output_shape__", None)
    if shape is None:
        fl = 0
    else:
        N, C_out, H_out, W_out = shape
        C_in = self.in_channels
        kh, kw = self.kernel_size
        # each output element: C_in*kh*kw multiplications + (C_in*kh*kw - 1) adds ~ 2*ops
        flops_per_element = C_in * kh * kw * 2
        fl = N * C_out * H_out * W_out * flops_per_element
    # record on the module
    self.__flops__ = fl
    return fl


Conv2d.accumulate_flops = _conv2d_accumulate_flops

# --- Monkey-patch Conv2d to support parameter counting for FLOPs counter ---
# Ensure a default __params__ attribute
Conv2d.__params__ = 0  # default


def _conv2d_accumulate_params(self):
    """
    Compute and record number of parameters for this Conv2d.
    """
    p = self.weight.numel()
    if self.bias is not None:
        p += self.bias.numel()
    self.__params__ = p
    return p


Conv2d.accumulate_params = _conv2d_accumulate_params
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.utils import to_dense_batch


class AttentionConvLayer(nn.Module):
    """
    Convolves over the time dimension of attention scores, preserving the key dimension.
    Input: [batch, heads, tgt_len, src_len]
    Output: [batch, heads, tgt_len, src_len]
    """

    def __init__(
        self,
        proj_dim: int,
        filter_heights: list = [1, 3, 5],
        vertical_stride: int = 1,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.filter_heights = filter_heights
        self.vertical_stride = vertical_stride
        self.convs = nn.ModuleList()
        for h in self.filter_heights:
            pad_h = h // 2
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(h, self.proj_dim),
                stride=(self.vertical_stride, 1),
                padding=(pad_h, 0),
                bias=False,
            )

            # attach FLOPs counter to this conv2d
            def _accumulate_flops_conv(self_conv):
                return getattr(self_conv, "__last_flops__", 0)

            conv.accumulate_flops = _accumulate_flops_conv.__get__(conv, conv.__class__)
            self.convs.append(conv)

    def forward(self, x: Tensor) -> Tensor:
        B, H, T, S = x.size()
        assert (
            S == self.proj_dim
        ), f"AttentionConvLayer expects width R == proj_dim ({self.proj_dim}), got {S}"
        # reshape into 4D for conv2d: [B*H,1,T,S]
        x_ = x.view(B * H, 1, T, S)
        # apply conv(s)
        if len(self.convs) == 1:
            conv_out = self.convs[0](x_)
        else:
            outs = [conv(x_) for conv in self.convs]
            conv_out = torch.stack(outs, dim=-1).mean(dim=-1)
        out = conv_out.squeeze(1)
        out = out.expand(-1, -1, S)  # [B*H, T', S]  <-- restore width

        newT = out.size(1)
        if self.vertical_stride > 1 and newT != T:
            out = F.interpolate(
                out.unsqueeze(1), size=(T, S), mode="bilinear", align_corners=False
            ).squeeze(1)
        else:
            out = out.contiguous()

        return out.view(B, H, T, S)


class MultiheadLinearAttention(nn.Module):
    """
    Multi-headed Linformer-style attention with optional clustering
    and convolution.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        compressed: int = 8,
        max_seq_len: int = 128,
        shared_kv_compressed: bool = False,
        shared_compress_layer: Optional[nn.Linear] = None,
        proj_dim: Optional[int] = None,
        cluster_E: bool = True,
        cluster_F: bool = True,
        share_EF: bool = False,
        convolution: bool = True,
        conv_filter_heights: list = [1, 1, 3, 3, 5, 5, 7, 7, 9, 9],
        vertical_stride: int = 1,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
    ):
        super().__init__()
        # dims and flags
        self.embed_dim = embed_dim
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.dropout = dropout

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.shared_kv_compressed = shared_kv_compressed
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        # compression setup
        self.max_seq_len = max_seq_len
        base_len = max_seq_len // compressed
        self.proj_dim = proj_dim or base_len
        if shared_compress_layer is None:
            self.compress_k = nn.Linear(max_seq_len, self.proj_dim, bias=False)
            self.compress_v = nn.Linear(max_seq_len, self.proj_dim, bias=False)
        else:
            self.compress_k = shared_compress_layer
            self.compress_v = (
                shared_compress_layer
                if shared_kv_compressed
                else nn.Linear(max_seq_len, self.proj_dim, bias=False)
            )

        # QKV and out projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.E = nn.Parameter(
            torch.empty(self.num_heads, self.max_seq_len, self.proj_dim)
        )
        if share_EF:
            self.F = self.E
        else:
            self.F = nn.Parameter(
                torch.empty(self.num_heads, self.max_seq_len, self.proj_dim)
            )

        # clustering parameters
        self.cluster_E = cluster_E
        self.cluster_F = cluster_F
        self.share_EF = share_EF
        if cluster_E:
            self.chunk_size_E = math.ceil(max_seq_len / self.proj_dim)
            self.cluster_E_W = Parameter(
                torch.Tensor(num_heads, self.proj_dim, self.chunk_size_E)
            )
        if share_EF:
            self.cluster_F_W = self.cluster_E_W
        elif cluster_F:
            self.chunk_size_F = math.ceil(max_seq_len / self.proj_dim)
            self.cluster_F_W = Parameter(
                torch.Tensor(num_heads, self.proj_dim, self.chunk_size_F)
            )

        # convolution over scores
        self.convolution = convolution
        if convolution:
            self.attn_conv = AttentionConvLayer(
                self.proj_dim, conv_filter_heights, vertical_stride
            )

        # optional biases
        if add_bias_kv:
            self.bias_k = Parameter(torch.zeros(1, 1, embed_dim))
            self.bias_v = Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.bias_k = None
            self.bias_v = None

        self.reset_parameters()

    def reset_parameters(self):
        for p in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(p.weight)
            if p.bias is not None:
                nn.init.constant_(p.bias, 0.0)
                # Init per-head E/F like TF (Glorot)
        scale_len = 1.0 / math.sqrt(self.max_seq_len)
        nn.init.uniform_(
            self.E, -scale_len, scale_len
        )  # CHANGED: explicit per-head init
        if isinstance(self.F, nn.Parameter) and self.F.data_ptr() != self.E.data_ptr():
            nn.init.uniform_(self.F, -scale_len, scale_len)
        nn.init.xavier_uniform_(self.compress_k.weight)
        nn.init.xavier_uniform_(self.compress_v.weight)
        if hasattr(self, "cluster_E_W"):
            nn.init.xavier_uniform_(self.cluster_E_W)
        if hasattr(self, "cluster_F_W") and self.cluster_F_W is not self.cluster_E_W:
            nn.init.xavier_uniform_(self.cluster_F_W)
        if self.add_bias_kv:
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)

    def _project_classic_per_head(self, x: Tensor, P: Tensor, seq_len: int) -> Tensor:
        """
        x: [N, B, embed_dim]  (N = seq_len)
        P: [H, N_max, R]  per-head projection matrix
        return: [B, H, R, D]
        """
        B = x.size(1)
        H, D = self.num_heads, self.head_dim
        x_ = x.permute(1, 2, 0)  # [B, embed_dim, N]
        x_ = x_.view(B, H, D, seq_len).permute(0, 1, 3, 2)  # [B,H,N,D]
        P_ = P[:, :seq_len, :]  # [H,N,R]
        out = torch.einsum("b h n d, h n r -> b h r d", x_, P_)  # [B,H,R,D]
        return out

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        T, B, _ = query.size()
        q = self.q_proj(query)

        # select K/V
        if self.self_attention and (key is None and value is None):
            k_input, v_input = query, query
        else:
            k_input, v_input = key, value
        S = k_input.size(0)

        # compute K
        if self.cluster_E:
            kproj = self.k_proj(k_input)
            kproj = kproj.view(S, B, self.num_heads, self.head_dim)
            kproj = kproj.permute(1, 2, 0, 3)
            pad_e = self.chunk_size_E * self.proj_dim - kproj.size(2)
            kp = F.pad(kproj, (0, 0, 0, pad_e))
            kp = kp.view(
                B, self.num_heads, self.proj_dim, self.chunk_size_E, self.head_dim
            )
            k = torch.einsum("b h p c d, h p c -> b h p d", kp, self.cluster_E_W)
        else:
            k = self._project_classic_per_head(k_input, self.E, S)  # [B,H,S,D]

        # compute V
        if self.cluster_F:
            vproj = self.v_proj(v_input)
            vproj = vproj.view(S, B, self.num_heads, self.head_dim)
            vproj = vproj.permute(1, 2, 0, 3)
            pad_f = self.chunk_size_F * self.proj_dim - vproj.size(2)
            vp = F.pad(vproj, (0, 0, 0, pad_f))
            vp = vp.view(
                B, self.num_heads, self.proj_dim, self.chunk_size_F, self.head_dim
            )
            v = torch.einsum("b h p c d, h p c -> b h p d", vp, self.cluster_F_W)
        else:
            v = self._project_classic_per_head(v_input, self.F, S)  # [B,H,S,D]

        # reshape for attention
        q = q.view(T, B, self.num_heads, self.head_dim)
        q = q.permute(1, 2, 0, 3).reshape(B * self.num_heads, T, self.head_dim)
        k = k.reshape(B * self.num_heads, -1, self.head_dim)
        v = v.reshape(B * self.num_heads, -1, self.head_dim)
        q = q * self.scaling

        # attention scores
        scores = torch.bmm(q, k.transpose(1, 2))
        if self.convolution:
            scores = scores.view(B, self.num_heads, T, -1)
            scores = self.attn_conv(scores)
            scores = scores.reshape(B * self.num_heads, T, -1)

        # add biases
        if self.add_bias_kv:
            bias = self.bias_k.repeat(B * self.num_heads, 1, 1)
            scores = torch.cat([scores, bias], dim=2)

        # apply attn_mask
        if attn_mask is not None:
            BpH, Tp, Pp = attn_mask.shape
            mask_flat = attn_mask.reshape(-1, Pp)
            comp_w = self.compress_k.weight[:, :Pp]
            mask_comp = F.linear(mask_flat, comp_w)
            mask_comp = mask_comp.view(BpH, Tp, self.proj_dim)
            scores = scores + mask_comp

        # apply padding mask
        if key_padding_mask is not None:
            P_orig = key_padding_mask.size(1)
            fill = min(P_orig, self.proj_dim)
            mask_cp = torch.ones(
                B, self.proj_dim, device=key_padding_mask.device, dtype=torch.bool
            )
            mask_cp[:, :fill] = key_padding_mask[:, :fill]
            mask4d = mask_cp.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, T, -1)
            scores = scores.view(B, self.num_heads, T, -1)
            scores = scores.masked_fill(mask4d, float("-inf"))
            scores = scores.reshape(B * self.num_heads, T, -1)

        # zero-attn
        if self.add_zero_attn:
            zeros = scores.new_zeros(scores.size(0), scores.size(1), 1)
            scores = torch.cat([scores, zeros], dim=2)

        # softmax
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # attend
        out = torch.bmm(attn_probs, v)
        out = out.view(B, self.num_heads, T, self.head_dim)
        out = out.permute(2, 0, 1, 3).reshape(T, B, self.embed_dim)
        out = self.out_proj(out)

        weights = (
            attn_probs.view(B, self.num_heads, T, -1).mean(dim=1)
            if need_weights
            else None
        )
        return out, weights


class MCABlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.1
    ):
        super().__init__()
        self.attn = MultiheadLinearAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            self_attention=True,
            convolution=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x_dense: Tensor, pad_mask: Tensor) -> Tensor:
        # x_dense: [B, T, D]; pad_mask: [B, T] True means padded
        q = x_dense.permute(1, 0, 2)
        attn_out, _ = self.attn(
            q, None, None, key_padding_mask=pad_mask, need_weights=False
        )
        attn_out = attn_out.permute(1, 0, 2)
        x = self.norm1(x_dense + self.dropout(attn_out))
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


class TrackingMultiConvEncoder(nn.Module):
    """
    Wrapper that applies MultiheadLinearAttention blocks to PyG tracking graphs.
    Produces per-node embeddings compatible with the tracking trainer.
    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int = 24,
        num_heads: int = 8,
        n_layers: int = 4,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )

        self.blocks = nn.ModuleList(
            [
                MCABlock(
                    embed_dim=h_dim,
                    num_heads=num_heads,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, data) -> Tensor:
        # Accept PyG Data or dict-like
        if isinstance(data, dict):
            x, batch = data["x"], data["batch"]
        else:
            x, batch = data.x, data.batch

        x = self.feat_encoder(x)
        x_dense, mask = to_dense_batch(x, batch)  # [B, T, D], [B, T]
        pad_mask = ~mask  # True where padded

        for blk in self.blocks:
            x_dense = blk(x_dense, pad_mask)

        # Flatten back to [N, D]
        out = x_dense[mask]
        return out
