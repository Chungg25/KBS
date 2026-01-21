import torch
from torch import nn
from layers.revin import RevIN
import torch.nn.functional as F

class MultiScaleDilatedConv(nn.Module):
    """Multi-scale dilated convolutions for capturing global temporal patterns"""
    def __init__(self, d_model: int, kernel_size: int = 3):
        super(MultiScaleDilatedConv, self).__init__()
        # Different dilation rates to capture different temporal scales
        self.conv_d1 = nn.Conv1d(d_model, d_model, kernel_size, dilation=1, padding=1)
        self.conv_d2 = nn.Conv1d(d_model, d_model, kernel_size, dilation=2, padding=2)
        self.conv_d4 = nn.Conv1d(d_model, d_model, kernel_size, dilation=4, padding=4)
        self.conv_d8 = nn.Conv1d(d_model, d_model, kernel_size, dilation=8, padding=8)
        
        # Fusion layer
        self.fusion = nn.Conv1d(d_model * 4, d_model, 1)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_model, seq_len]
        out1 = self.act(self.conv_d1(x))
        out2 = self.act(self.conv_d2(x))
        out4 = self.act(self.conv_d4(x))
        out8 = self.act(self.conv_d8(x))
        
        # Concatenate multi-scale features
        out = torch.cat([out1, out2, out4, out8], dim=1)
        out = self.fusion(out)
        out = self.act(out)
        
        return out

class GlobalContextModule(nn.Module):
    """Global context aggregation using pooling operations"""
    def __init__(self, d_model: int):
        super(GlobalContextModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Transform global context back to sequence
        self.fc1 = nn.Linear(d_model * 2, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_model, seq_len]
        B, C, L = x.shape
        
        # Extract global context
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # [B, d_model]
        max_pool = self.global_max_pool(x).squeeze(-1)  # [B, d_model]
        
        # Combine global information
        global_context = torch.cat([avg_pool, max_pool], dim=1)  # [B, d_model*2]
        global_context = self.fc1(global_context)
        global_context = self.act(global_context)
        global_context = self.fc2(global_context)  # [B, d_model]
        
        # Broadcast back to sequence length
        global_context = global_context.unsqueeze(-1).expand(-1, -1, L)  # [B, d_model, seq_len]
        
        return global_context

class GatedTemporalCausality(nn.Module):
    def __init__(self, c_in: int, d_model: int, kernel_size: int):
        super(GatedTemporalCausality, self).__init__()
        # Temporal Causality
        self.conv1d = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size
        )

        self.pad = kernel_size - 1
        self.conv1 = nn.Conv1d(d_model, d_model, 1)
        # self.dropout = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pad(x, (self.pad, 0))
        s = self.conv1d(h)
        gate = self.sig(self.conv1(s))
        s = s * gate
        s = self.act(s)
        # s = self.dropout(s)
        return s

class PeriodMixing(nn.Module):
    def __init__(self, c_in: int, seq_len: int, pred_len: int, period_len: int, d_model: int):
        super(PeriodMixing, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        self.d_model = d_model
        self.c_in = c_in
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        self.lin1 = nn.Linear(self.seg_num_x, self.d_model * 2)
        self.lin2 = nn.Linear(self.d_model * 2, self.seg_num_y)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.lin1(y)
        y = self.gelu(y)
        y = self.lin2(y)
        y = y.permute(0, 2, 1).reshape(-1, self.c_in, self.pred_len)
        return y

class SeasonalModule(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in
        self.period_len = period_len
        self.d_model = d_model

        kernel_size = 2
        self.W = nn.Conv1d(d_model, d_model * 2, 1)
        self.W2 = nn.Conv1d(d_model * 2, c_in, 1)

        # Local temporal features
        self.gate_tcn = GatedTemporalCausality(c_in, d_model, kernel_size)
        
        # Global temporal features
        self.multi_scale_conv = MultiScaleDilatedConv(d_model, kernel_size=3)
        self.global_context = GlobalContextModule(d_model)
        
        # Fusion of local and global features
        self.fusion = nn.Conv1d(d_model * 3, d_model, 1)
        
        self.period_mixing = PeriodMixing(c_in, seq_len, pred_len, period_len, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.revin_layer = RevIN(c_in, affine=True)

    def forward(self, s):
        # s: [B, seq_len, C]
        s = self.revin_layer(s, 'norm')
        s = s.permute(0, 2, 1)  # [B, C, seq_len]
        B, C, _ = s.shape

        # Local temporal features (gated convolution)
        s_local = self.gate_tcn(s)  # [B, d_model, seq_len]
        
        # Multi-scale global features (dilated convolutions)
        s_multi_scale = self.multi_scale_conv(s_local)  # [B, d_model, seq_len]
        
        # Global context features (pooling-based)
        s_global = self.global_context(s_local)  # [B, d_model, seq_len]
        
        # Fuse all features: local + multi-scale + global context
        s = torch.cat([s_local, s_multi_scale, s_global], dim=1)  # [B, d_model*3, seq_len]
        s = self.fusion(s)  # [B, d_model, seq_len]
        s = self.act(s)
        s = self.dropout(s)

        s = self.W(s)
        s = self.act(s)
        s = self.dropout(s)
        s = self.W2(s)
        s = self.act(s)
        s = self.dropout(s)
        
        s = self.period_mixing(s)
        y = s.permute(0, 2, 1)  # [B, pred_len, C]
        y = self.revin_layer(y, "denorm")
        return y

class TrendModule(nn.Module):
    def __init__(self, c_in, seq_len, pred_len):
        super().__init__()
        self.fc5 = nn.Linear(seq_len, seq_len * 2)
        self.gelu1 = nn.GELU()
        self.fc7 = nn.Linear(seq_len * 2, seq_len)
        self.fc8 = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(0.3)
        self.revin_layer = RevIN(c_in, affine=True)

    def forward(self, t):
        # t: [B, seq_len, C]
        t = self.revin_layer(t, "norm")
        t = t.permute(0, 2, 1)  # [B, C, seq_len]
        B, C, _ = t.shape
        # pass data qua MLP thong thuong
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.dropout(t)
        t = self.fc7(t)
        t = self.gelu1(t)
        t = self.dropout(t)
        t = self.fc8(t)

        t = t.permute(0, 2, 1)  # [B, pred_len, C]
        t = self.revin_layer(t, "denorm")
        return t


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout=0.1):
        super().__init__()
        self.non_linear = SeasonalModule(seq_len, pred_len, c_in, period_len, d_model, dropout)
        self.linear = TrendModule(c_in, seq_len, pred_len)

    def forward(self, s, t):
        y_non_linear = self.non_linear(s)
        y_linear = self.linear(t)
        return y_linear + y_non_linear