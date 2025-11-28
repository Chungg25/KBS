
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

        self.W1 = nn.Linear(seq_len, pred_len)
        self.W = nn.Conv1d(d_model, d_model * 2, 1)
        self.W2 = nn.Conv1d(d_model * 2, c_in, 1)

        kernel_size = 2
        self.pad = kernel_size - 1

        # Temporal Causality
        self.conv1d = nn.Conv1d(
            in_channels=c_in,
            out_channels=self.d_model,
            kernel_size=kernel_size)
        
        # self.gate = nn.Sequential(
        #     nn.Conv1d(self.d_model, self.d_model, 1),
        #     nn.Sigmoid()
        # )

        self.conv1 = nn.Conv1d(self.d_model, self.d_model, 1, bias=True)
        self.sig = nn.Sigmoid()

        self.act = nn.GELU()

        self.revin_layer = RevIN(c_in, affine=True)

        self.mlp = nn.Sequential( 
            nn.Linear(self.seq_len // self.period_len, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.pred_len // self.period_len)
        )

    def forward(self, s):
        # s: [B, seq_len, C]
        s = self.revin_layer(s, 'norm')
        s = s.permute(0, 2, 1)  # [B, C, seq_len]
        B, C, _ = s.shape

        # Padding de dam bao output = input
        h = F.pad(s, (self.pad, 0))
        s = self.conv1d(h)
        # gate_val = self.gate(s)

        # s = self.conv1(s)
        gate_val = self.sig(self.conv1(s))
        s = s * gate_val
        s = self.act(s)

        s = F.dropout(s, 0.3, self.training)

        s = self.W(s)
        s = self.act(s)
        
        s = F.dropout(s, 0.3, self.training)
        s = self.W2(s)
        s = self.act(s)

        s = F.dropout(s, 0.3, self.training)

        # print(s.shape)

        # s = self.mlp_global(s)

        seg_num_x = self.seq_len // self.period_len
        seg_num_y = self.pred_len // self.period_len
        
        s = s.reshape(-1, seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.c_in, self.pred_len)

        # y = self.W1(s)
        y = y.permute(0, 2, 1)  # [B, pred_len, C]
        y = self.revin_layer(y, "denorm")
        return y

class LinearStream(nn.Module):
    def __init__(self, c_in, seq_len, pred_len):
        super().__init__()
        self.fc5 = nn.Linear(seq_len, seq_len * 2)
        self.gelu1 = nn.GELU()
        self.fc7 = nn.Linear(seq_len * 2, seq_len)
        self.fc8 = nn.Linear(seq_len, pred_len)
        self.revin_layer = RevIN(c_in, affine=True)

    def forward(self, t):
        # t: [B, seq_len, C]
        t = self.revin_layer(t, "norm")
        t = t.permute(0, 2, 1)  # [B, C, seq_len]
        B, C, _ = t.shape
        # pass data qua MLP thong thuong
        t = self.fc5(t)
        t = self.gelu1(t)
        t = F.dropout(t, 0.3, self.training)
        t = self.fc7(t)
        t = self.gelu1(t)
        t = F.dropout(t, 0.3, self.training)
        t = self.fc8(t)

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
    
