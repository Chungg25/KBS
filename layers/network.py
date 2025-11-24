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

        self.W = nn.Linear(c_in, d_model)
        self.W1 = nn.Linear(d_model, c_in)

        kernel_size = period_len
        self.pad = kernel_size - 1

        # Temporal Causality
        self.conv1d = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=kernel_size,
            groups=self.d_model,
        )

        self.ln1 = nn.LayerNorm(d_model)
        
        # self.act = nn.GELU()
        self.act = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
        )

        self.mlp = nn.Sequential(
            nn.Linear(period_len, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, period_len)
        )
        # MLP cho tương tác giữa các segment
        self.segment_mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.seg_num_x * 2),
            nn.GELU(),
            nn.Linear(self.seg_num_x * 2, self.seg_num_y)
        )

        self.revin_layer = RevIN(d_model,affine=True,subtract_last=False)

    def forward(self, s):
        # s: [B, seq_len, C]
        s = self.W(s)
        s = self.revin_layer(s, 'norm')
        s = s.permute(0, 2, 1)  # [B, C, seq_len]
        B, C, _ = s.shape

        # Padding de dam bao output = input
        h = F.pad(s, (self.pad, 0))
        s = self.conv1d(h)
        s = self.ln1(s)
        s = self.act(s)

        # Học đặc trưng bên trong từng segment
        s = s.reshape(B, self.seg_num_x, self.period_len)
        s = self.mlp(s)
        # [B, seg_num_x, period_len]

        # Học tương tác giữa các segment
        s = s.permute(0, 2, 1)  # [B, period_len, seg_num_x]
        s = self.segment_mlp(s)
        # [B, period_len, seg_num_y]
        s = s.permute(0, 2, 1)  # [B, seg_num_y, period_len]
        s = s.reshape(B, -1, self.d_model)  # [B, pred_len, d_model]

        y = self.revin_layer(s, "denorm")
        y = self.W1(y)
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
        y_non_linear = self.non_linear(s)
        y_linear = self.linear(t)
        return y_linear + y_non_linear
