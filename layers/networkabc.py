
import torch
from torch import nn
from layers.revin import RevIN
import torch.nn.functional as F


class ChannelModule(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in
        self.period_len = period_len
        self.d_model = d_model
        self.dropout = dropout

        self.W = nn.Conv1d(d_model, d_model * 2, 1)
        self.W2 = nn.Conv1d(d_model * 2, c_in, 1)

        kernel_size = 2
        self.pad = kernel_size - 1

        self.conv1d = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=kernel_size,
            padding=0,
            groups=c_in)
        

        self.conv1 = nn.Conv1d(c_in, c_in, 1, bias=True)
        self.sig = nn.Sigmoid()

        self.global_conv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model, kernel_size=1
        )

        self.act = nn.GELU()


    def forward(self, s):
        s = s.permute(0, 2, 1)  # [B, C, seq_len]

        h = F.pad(s, (self.pad, 0))
        s = self.conv1d(h)

        gate_val = self.sig(self.conv1(s))
        s = s * gate_val
        s = self.act(s)

        s = self.global_conv(s)

        s = self.W(s)
        s = self.act(s)
        
        s = F.dropout(s, self.dropout, self.training)
        s = self.W2(s)
        s = self.act(s)

        s = F.dropout(s, self.dropout, self.training)

        return s

class LocalTemporal(nn.Module):
    def __init__(self, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            dilation=dilation,
            padding= (kernel_size - 1) // 2 * dilation,
            groups=1
        )

    def forward(self, x):
        return self.conv(x)

class NonLinearStream(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in
        self.period_len = period_len
        self.d_model = d_model

        self.channel = ChannelModule(seq_len, pred_len, c_in, period_len, d_model, dropout)

        self.revin_layer = RevIN(c_in, affine=True)

        self.LocalTemporal = LocalTemporal(kernel_size=3, dilation=1)

        self.seg_num_x = self.seq_len // self.period_len

        self.GlobalTemporal = nn.Sequential(
            nn.LayerNorm(self.seg_num_x),
            nn.Linear(self.seg_num_x, self.d_model*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model*2, self.seg_num_x)
        )

        self.local_gate = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential( 
            nn.Linear(self.seq_len // self.period_len, self.d_model*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model*2, self.pred_len // self.period_len)
        )

    def forward(self, s):
        # s: [B, seq_len, C]
        s = self.revin_layer(s, 'norm')
        B, S, C = s.shape

        s = self.channel(s)

        seg_num_x = self.seq_len // self.period_len
        seg_num_y = self.pred_len // self.period_len
        
        s = s.reshape(-1, seg_num_x, self.period_len)
        s = s.reshape(B*C*seg_num_x, 1, self.period_len)
        r = s 
        s = self.LocalTemporal(s)
        gate = self.local_gate(s)          # gate value ∈ (0,1)
        s = s * gate  
        s = s + r
        s = s.reshape(B*C, seg_num_x, self.period_len).permute(0, 2, 1)
        s = s + self.GlobalTemporal(s)
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.c_in, self.pred_len)

        # # y = self.W1(s)
        y = y.permute(0, 2, 1)  # [B, pred_len, C]
        y = self.revin_layer(y, "denorm")
        return y
    
    

class LinearStream(nn.Module):
    def __init__(self, c_in, seq_len, pred_len, kernel_size=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in

        self.revin_layer = RevIN(c_in, affine=True)

        # ===== Local modeling (depthwise conv) =====
        self.pad = kernel_size - 1
        self.local_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=kernel_size,
            groups=c_in
        )


        hidden = 4
        self.global_linear = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, hidden*2),
            nn.GELU(),
            # nn.Dropout(0.3),
            nn.Linear(hidden*2, seq_len),
            nn.LayerNorm(seq_len),
        )


        # ===== Prediction head =====
        # self.proj = nn.Linear(seq_len, pred_len)
        self.proj1 = nn.Linear(seq_len, pred_len//2)
        self.ln = nn.LayerNorm(pred_len //2)
        self.drop = nn.Dropout(0.3)
        self.proj2 = nn.Linear(pred_len//2, pred_len)

    def forward(self, t):
        # t: [B, seq_len, C]
        t = self.revin_layer(t, "norm")
        t = t.permute(0, 2, 1)  # [B, C, seq_len]
        
        # ===== Local =====
        h = F.pad(t, (self.pad, 0))
        t_local = self.local_conv(h)

        t = t + t_local

        t_global = self.global_linear(t)

        t = t + t_global

        # t = self.proj(t)

        t = self.proj1(t)
        t = self.ln(t)
        t = self.drop(t)
        t=self.proj2(t)

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
    