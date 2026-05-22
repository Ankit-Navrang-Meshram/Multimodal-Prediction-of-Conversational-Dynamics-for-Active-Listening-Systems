import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankQuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16):
        super(LowRankQuaternionLinear, self).__init__()
        self.in_d = in_features // 4
        self.out_d = out_features // 4
        self.rank = rank
        
        # Each quaternion component is now defined by TWO low-rank matrices
        # Standard: out_d * in_d parameters
        # Low-Rank: (out_d * rank) + (rank * in_d) parameters
        self.u_r = nn.Parameter(torch.randn(self.out_d, self.rank))
        self.v_r = nn.Parameter(torch.randn(self.rank, self.in_d))
        
        self.u_i = nn.Parameter(torch.randn(self.out_d, self.rank))
        self.v_i = nn.Parameter(torch.randn(self.rank, self.in_d))
        
        self.u_j = nn.Parameter(torch.randn(self.out_d, self.rank))
        self.v_j = nn.Parameter(torch.randn(self.rank, self.in_d))
        
        self.u_k = nn.Parameter(torch.randn(self.out_d, self.rank))
        self.v_k = nn.Parameter(torch.randn(self.rank, self.in_d))
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Proper initialization for low-rank products
        for p in [self.u_r, self.u_i, self.u_j, self.u_k]:
            nn.init.xavier_normal_(p)
        for p in [self.v_r, self.v_i, self.v_j, self.v_k]:
            nn.init.kaiming_uniform_(p)

    def get_weight(self, u, v):
        return torch.matmul(u, v)

    def forward(self, x):
        r, i, j, k = torch.chunk(x, 4, dim=1)
        
        # Reconstruct full weights on-the-fly (memory efficient)
        wr, wi, wj, wk = self.get_weight(self.u_r, self.v_r), \
                         self.get_weight(self.u_i, self.v_i), \
                         self.get_weight(self.u_j, self.v_j), \
                         self.get_weight(self.u_k, self.v_k)

        # Hamilton product logic
        cat_r = F.linear(r, wr) - F.linear(i, wi) - F.linear(j, wj) - F.linear(k, wk)
        cat_i = F.linear(r, wi) + F.linear(i, wr) + F.linear(j, wk) - F.linear(k, wj)
        cat_j = F.linear(r, wj) - F.linear(i, wk) + F.linear(j, wr) + F.linear(k, wi)
        cat_k = F.linear(r, wk) + F.linear(i, wj) - F.linear(j, wi) + F.linear(k, wr)
        
        return torch.cat([cat_r, cat_i, cat_j, cat_k], dim=-1) + self.bias


class DeepLowRankFusion(nn.Module):
    def __init__(self, feature_dim=256, rank=16, output_dim=3):
        super().__init__()
        self.dim = feature_dim
        self.global_context = nn.Parameter(torch.randn(1, self.dim) * 0.02)
        
        # Layer 1: Processing at full dimension
        self.layer1 = LowRankQuaternionLinear(self.dim * 4, self.dim * 4, rank=rank)
        self.act1 = nn.GELU()
        
        # Layer 2: Bottleneck (reduces to dim // 2)
        self.layer2 = LowRankQuaternionLinear(self.dim * 4, (self.dim // 2) * 4, rank=rank // 2)
        self.act2 = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        
        # Final prediction layer
        self.classifier = nn.Linear((self.dim // 2) * 4, output_dim)

    def forward(self, text, audio, video):
        batch_size = (text if text is not None else audio).size(0)
        device = (text if text is not None else audio).device
        
        # Handle modal dropout safely
        if text is None: text = torch.zeros(batch_size, self.dim, device=device)
        if audio is None: audio = torch.zeros(batch_size, self.dim, device=device)
        if video is None: video = torch.zeros(batch_size, self.dim, device=device)
        
        r = self.global_context.expand(batch_size, -1)
        q_in = torch.cat([r, text, audio, video], dim=1)
        
        # Forward through deep layers
        x = self.act1(self.layer1(q_in))
        x = self.act2(self.layer2(x))
        x = self.dropout(x)
        
        return self.classifier(x)


if __name__ == "__main__":
    QF = DeepLowRankFusion(rank=16)

    batch_size = 16
    Xt = torch.randn(batch_size, 256)
    Xa = torch.randn(batch_size, 256)
    Xv = torch.randn(batch_size, 256)

    print(QF(Xa, Xv, Xt).shape)
    print(QF(None, Xv, Xt).shape)
    print(QF(Xa, None, Xt).shape)
    print(QF(Xa, Xv, None).shape)

    params = sum(p.numel() for p in QF.parameters())
    print(f"  Parameters: {params:,}")
    print()
