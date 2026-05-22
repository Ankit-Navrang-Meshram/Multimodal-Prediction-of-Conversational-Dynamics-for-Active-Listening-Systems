import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_

class QuaternionLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuaternionLinear, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        
        # 4 distinct blocks to construct the Hamilton product matrix
        self.r_weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.i_weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.j_weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.k_weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        
        xavier_normal_(self.r_weight)
        xavier_normal_(self.i_weight)
        xavier_normal_(self.j_weight)
        xavier_normal_(self.k_weight)
        self.bias.data.fill_(0)

    def forward(self, x):
        # x shape: (batch_size, in_features * 4)
        r, i, j, k = torch.chunk(x, 4, dim=1)
        
        # Hamilton product logic
        cat_r = F.linear(r, self.r_weight) - F.linear(i, self.i_weight) - F.linear(j, self.j_weight) - F.linear(k, self.k_weight)
        cat_i = F.linear(r, self.i_weight) + F.linear(i, self.r_weight) + F.linear(j, self.k_weight) - F.linear(k, self.j_weight)
        cat_j = F.linear(r, self.j_weight) - F.linear(i, self.k_weight) + F.linear(j, self.r_weight) + F.linear(k, self.i_weight)
        cat_k = F.linear(r, self.k_weight) + F.linear(i, self.j_weight) - F.linear(j, self.i_weight) + F.linear(k, self.r_weight)
        
        out = torch.cat([cat_r, cat_i, cat_j, cat_k], dim=-1) + self.bias
        return out


class QuaternionFusion(nn.Module):
    def __init__(self, feature_dim=256, output_dim=3):
        super(QuaternionFusion, self).__init__()
        self.dim = feature_dim
        
        # Trainable global context acts as the real (r) component
        self.global_context = Parameter(torch.zeros(1, self.dim))
        nn.init.normal_(self.global_context, std=0.02)
        
        # 4 components of size `dim` = total input size of 4 * dim
        self.quat_linear = QuaternionLinear(self.dim * 4, self.dim * 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.dim * 4, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, text_x, audio_x, video_x):
        # Determine batch size and device from whichever modality is present
        temp_x = text_x if text_x is not None else (audio_x if audio_x is not None else video_x)
        batch_size = temp_x.size(0)
        device = temp_x.device
        
        # Zero-pad missing modalities (robust to your random_drop_modal_rate)
        if text_x is None: text_x = torch.zeros(batch_size, self.dim, device=device)
        if audio_x is None: audio_x = torch.zeros(batch_size, self.dim, device=device)
        if video_x is None: video_x = torch.zeros(batch_size, self.dim, device=device)

        r = self.global_context.expand(batch_size, -1)
        
        # Construct hypercomplex vector: r + text*i + audio*j + video*k
        q_input = torch.cat([r, text_x, audio_x, video_x], dim=1)
        
        q_out = self.quat_linear(q_input)
        out = self.classifier(q_out)
        
        return out


if __name__ == "__main__":
    QF = QuaternionFusion()

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