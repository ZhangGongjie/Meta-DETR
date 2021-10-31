import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class SingleHeadSiameseAttention(nn.Module):
    """ Single-Head Attention Module. Weights for Q and K are shared in a Siamese manner. No proj weights for V."""
    def __init__(self, d_model):
        super().__init__()
        self.n_head = 1
        self.d_model = d_model
        self.w_qk = nn.Linear(self.d_model, self.n_head * self.d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_model, 0.5))
        nn.init.normal_(self.w_qk.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_model)))

        self.dummy = nn.Parameter(torch.Tensor(1, self.d_model))
        nn.init.normal_(self.dummy)

        self.linear1 = nn.Sequential(nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(self.d_model, self.d_model // 2), nn.ReLU(inplace=True))
        self.linear3 = nn.Linear(self.d_model * 2, self.d_model)

    def forward(self, q, k, v, tsp):
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qk(q).view(sz_b, len_q, self.n_head, self.d_model)
        k = self.w_qk(k).view(sz_b, len_k, self.n_head, self.d_model)
        v = v.view(sz_b, len_v, self.n_head, self.d_model)

        tsp = tsp.view(sz_b, len_v, self.n_head, self.d_model)

        dummy = self.dummy.reshape(1, 1, 1, self.d_model).expand(sz_b, -1, self.n_head, -1)
        dummy_v = torch.zeros(sz_b, 1, self.n_head, self.d_model, device=v.device)

        k = torch.cat([k, dummy], dim=1)
        v = torch.cat([v, dummy_v], dim=1)
        tsp = torch.cat([tsp, dummy_v], dim=1)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_model)  # (n_head * b) x lq x d_model
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k + 1, self.d_model)  # (n_head * b) x lk x d_model
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1, self.d_model)  # (n_head * b) x lv x d_model
        tsp = tsp.permute(2, 0, 1, 3).contiguous().view(-1, len_v + 1, self.d_model)  # (n_head * b) x lv x d_model

        output, attn, log_attn = self.attention(q, k, v)
        tsp, _, _ = self.attention(q, k, tsp)

        output = output.view(self.n_head, sz_b, len_q, self.d_model)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n_head * d_model)

        tsp = tsp.view(self.n_head, sz_b, len_q, self.d_model)
        tsp = tsp.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n_head * d_model)

        output1 = self.linear1(output * residual)
        output2 = self.linear2(residual - output)
        output = self.linear3(
            torch.cat([output1, output2, residual], dim=2)
        )

        return output, tsp
