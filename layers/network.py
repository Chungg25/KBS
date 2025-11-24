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
            nn.Linear(self.d_model * 2, period_len),
        )

        self.W2 = nn.Linear(seq_len, pred_len)

        self.revin_layer = RevIN(d_model,affine=True,subtract_last=False)

    def forward(self, s):
        # s: [B, seq_len, C]
        s = self.W(s) # [B, seq_len, d_model]
        s = self.revin_layer(s, 'norm')
        s = s.permute(0, 2, 1)  # [B, d_model, seq_len]
        B, _, _ = s.shape

        # Padding de dam bao output = input
        h = F.pad(s, (self.pad, 0))  # [B, d_model, seq_len + pad]
        s = self.conv1d(h) # [B, d_model, seq_len]
        s = s.permute(0, 2, 1)  # [B, seq_len, d_model]
        s = self.ln1(s)
        s = s.permute(0, 2, 1)  # [B, d_model, seq_len]
        # print("s shape after conv1d:", s.shape)
        s = self.act(s)

        # s = s.reshape(-1, self.seg_num_x, self.period_len) # [B * d_model, seg_num_x, period_len]

        # # print("s shape before mlp:", s.shape)
        # y = self.mlp(s) # [B * d_model, seg_num_x, period_len]
        # y = y.permute(0, 2, 1)  # [B * d_model, period_len, seg_num_x]
        # y = y.reshape(B, self.d_model, self.period_len * self.seg_num_x) # [B, d_model, period_len]
        # # print("y shape after mlp:", y.shape)
        # y = self.W2(y)  # [B, d_model, pred_len]
        # y = y.permute(0, 2, 1)

        s = s.permute(0, 2, 1)  # [B, pred_len, d_model]
        # print("y shape before denorm:", y.shape)
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