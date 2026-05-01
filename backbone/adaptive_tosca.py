import torch
import torch.nn as nn
import torch.nn.init as init
from backbone.vit_tosca import Tosca


class ToscaAdaptiveViT(nn.Module):
    """
    ViT (with adapters) → W_rand projection → Tosca module.
    Tosca operates in the M-dim RP space, not the 768-dim feature space.
    """

    def __init__(self, model, mlp_ratio, se_ratio, flow, M=10000, embed_dim=768):
        super().__init__()
        self.vit = model
        self.embed_dim = embed_dim
        self.M = M

        for p in self.vit.parameters():
            p.requires_grad = False
        for name, p in self.vit.named_parameters():
            if "adaptmlp" in name:
                p.requires_grad = True

        self.W_rand = nn.Parameter(torch.randn(embed_dim, M))

        self.tosca = Tosca(M, mlp_ratio, se_ratio, flow)

    def forward_features(self, x):
        return self.vit.forward_features(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = torch.relu(x @ self.W_rand)
        x = self.tosca(x)
        return x

    def freeze_adapters(self):
        for name, p in self.vit.named_parameters():
            if "adaptmlp" in name:
                p.requires_grad = False

    def freeze_w_rand(self):
        self.W_rand.requires_grad = False

    def reset_tosca(self):
        for m in self.tosca.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)
