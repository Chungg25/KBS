import torch
from torch import nn
from layers.revin import RevIN
import torch.nn.functional as F

class MixerBlock(nn.Module):
    def __init__(self, channel, seq_len, d_model, dropout=0.1, expansion=2):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, seq_len)
        )

    def forward(self, x):
        # x: [B, C, seq_len]
        x_norm = self.norm(x) 
        z = self.mlp(x_norm)  
        out = x + z        
        return out

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

        self.W = nn.Linear(c_in, d_model)
        self.W1 = nn.Linear(d_model, c_in)

        kernel_size = period_len
        self.pad = kernel_size + 1

        # Temporal Causality
        self.conv1d = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=kernel_size,
        )

        self.ln1 = nn.LayerNorm(d_model)
        # self.act = nn.GELU()
        self.act = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
        )

        self.mixer = MixerBlock(channel=self.d_model, seq_len=self.seq_len, d_model=self.d_model, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(period_len, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, period_len)
        )

        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)

    def forward(self, s):
        # s: [B, seq_len, C]
        
        s = self.revin_layer(s, 'norm')
        s = self.W(s) # [B, seq_len, d_model]
        # s = self.revin_layer(s, 'norm')
        
        s = s.permute(0, 2, 1)  # [B, d_model, seq_len]
        B, _, _ = s.shape

        s = self.mixer(s) 

        # Padding de dam bao output = input
        h = F.pad(s, (self.pad, 0)) # [B, d_model, seq_len + pad]
        s = self.conv1d(h) # [B, d_model, seq_len]
        s = s.permute(0, 2, 1) 
        s = self.ln1(s) # [B, d_model, seq_len]
        s = s.permute(0, 2, 1) 
        s = self.act(s) # [B, d_model, seq_len]
        

        s = s.reshape(-1, self.seg_num_x, self.period_len)
        y = self.mlp(s)
        y = y.reshape(-1, self.d_model, self.period_len)
        # y = y.permute(0, 2, 1)

        y = s.permute(0, 2, 1)  # [B, pred_len, C]
        y = self.W1(y)
        y = self.revin_layer(y, "denorm")
        # y = self.W1(y)
        return y

class LinearStream(nn.Module):
    def __init__(self, c_in, seq_len, pred_len):
        super().__init__()
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.ln2 = nn.LayerNorm(pred_len)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)

    def forward(self, t):
        # t: [B, seq_len, C]
        t = self.revin_layer(t, "norm")
        t = t.permute(0, 2, 1)  # [B, C, seq_len]
        B, C, _ = t.shape
        # t = t.reshape(B * C, -1)

        # pass data qua MLP thong thuong
        t = self.fc5(t)
        t = self.ln1(t)
        t = self.gelu1(t)
        t = self.fc7(t)
        t = self.ln2(t)
        t = self.gelu1(t)
        t = self.fc8(t)

        # t = t.reshape(B, C, -1)
        t = t.permute(0, 2, 1)  # [B, pred_len, C]
        t = self.revin_layer(t, "denorm")
        return t

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout=0.1):
        super().__init__()
        self.non_linear = NonLinearStream(seq_len, pred_len, c_in, period_len, d_model, dropout)
        self.linear = LinearStream(c_in, seq_len, pred_len)

    def forward(self, s, t):
        # s: seasonal [B, seq_len, C]
        # t: trend [B, seq_len, C]
        y_non_linear = self.non_linear(s)
        y_linear = self.linear(t)
        return y_linear + y_non_linear
