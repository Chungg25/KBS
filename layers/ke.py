import torch
from torch import nn
from layers.revin import RevIN
import torch.nn.functional as F

class NonLinearStream(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in
        self.period_len = period_len
        self.d_model = d_model

        self.seg_num_x = seq_len // period_len
        self.seg_num_y = pred_len // period_len

        # self.W = nn.Linear(c_in, d_model)
        self.W1 = nn.Linear(seq_len, pred_len)

        kernel_size = period_len
        self.pad = kernel_size - 1

        # Temporal Causality
        self.conv1d = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=kernel_size,
        )

        self.ln1 = nn.LayerNorm(c_in)
        self.act = nn.GELU()

        # self.mlp = nn.Sequential(
        #     nn.Linear(period_len, self.d_model * 2),
        #     nn.GELU(),
        #     nn.Linear(self.d_model * 2, period_len)
        # )

        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)

    def forward(self, s):
        # s: [B, seq_len, C]
        s = self.revin_layer(s, 'norm')
        s = s.permute(0, 2, 1)  # [B, C, seq_len]
        B, C, _ = s.shape

        # Padding de dam bao output = input
        h = F.pad(s, (self.pad, 0))
        s = self.conv1d(h)
        s = self.ln1(s.transpose(-1, 1)).transpose(-1, 1)
        s = self.act(s)

        # s = s.reshape(-1, self.seg_num_x, self.period_len)
        # y = self.mlp(s)
        # y = y.reshape(-1, self.c_in, self.period_len)
        # y = y.permute(0, 2, 1)

        y = self.W1(s)
        y = y.permute(0, 2, 1)  # [B, pred_len, C]
        y = self.revin_layer(y, "denorm")
        return y

class TemporalDilatedCNN(nn.Module):
    """Simple stack of dilated 1D convolutions operating over the time dimension.
    
    
    Input shape: [B, C, T]
    Output shape: [B, C, T] (same temporal length)
    """
    def __init__(self, channels, kernel_size=3, dilations=(1, 2, 4, 8), dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        padding = lambda d: (kernel_size - 1) * d // 2 # keep same length
        for d in dilations:
            # keep in/out channels the same so we can use residuals easily
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=d,
                padding=padding(d),
            )
            self.layers.append(nn.Sequential(conv, nn.GELU(), nn.Dropout(dropout)))
        
        
        # a small 1x1 conv to mix channels if desired
        self.project = nn.Conv1d(channels, channels, kernel_size=1)
    
    
    def forward(self, x):
        # x: [B, C, T]
        residual = x
        for layer in self.layers:
            out = layer(x)
            # simple residual connection
            x = out + x
        x = self.project(x)
        return x + residual

class LinearStream(nn.Module):
    """Replaces the MLP temporal block with a temporal dilated CNN.
    
    
    Original signature preserved:
    c_in: number of channels/features (C)
    seq_len: input sequence length (T)
    pred_len: output sequence length (T_out)
    
    
    Forward expects input t: [B, seq_len, C]
    Returns: [B, pred_len, C]
    """
    def __init__(self, c_in, seq_len, pred_len, dilations=(1, 2, 4, 8), dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)
        
        
        # temporal dilated CNN works on shape [B, C, seq_len]
        self.temporal_cnn = TemporalDilatedCNN(c_in, kernel_size=3, dilations=dilations, dropout=dropout)
        
        
        # a tiny bottleneck / projection: map temporal dimension seq_len -> pred_len
        # we keep the same approach as original: a linear acting on the temporal dimension
        self.temporal_proj = nn.Linear(seq_len, pred_len)
        
        
        # optional layer norms
        self.ln_chan = nn.LayerNorm(c_in) # applied after permuting back to [B, T, C]
    
    
    def forward(self, t):
        # t: [B, seq_len, C]
        t = self.revin_layer(t, "norm")
        t = t.permute(0, 2, 1)
        t = self.temporal_cnn(t) # [B, C, seq_len]
        t = t.permute(0, 2, 1) # [B, seq_len, C]
        t = self.ln_chan(t)
        t = t.permute(0, 2, 1) # [B, C, seq_len]
        t = self.temporal_proj(t) # Linear(seq_len -> pred_len) applied to last dim
        # back to [B, pred_len, C]
        t = t.permute(0, 2, 1)
        t = self.revin_layer(t, "denorm")
        return t

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout=0.1):
        super().__init__()
        self.non_linear = NonLinearStream(seq_len, pred_len, c_in, period_len, d_model, dropout)
        self.linear = LinearStream(c_in, seq_len, pred_len)

    def forward(self, s, t):
        y_non_linear = self.non_linear(s)
        y_linear = self.linear(t)
        return y_linear + y_non_linear
