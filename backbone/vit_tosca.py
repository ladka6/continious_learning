import torch
import torch.nn as nn
import torch.nn.init as init

class Tosca(nn.Module):
    def __init__(self, dim, mlp_ratio=16, se_ratio=16, flow='tosca'):
        super().__init__()
        self.flow = flow
        self.norm = nn.LayerNorm(dim)
        
        # MLP part
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // mlp_ratio),
            nn.GELU(),
            nn.Linear(dim // mlp_ratio, dim)
        )
        
        # SE part
        self.se = nn.Sequential(
            nn.Linear(dim, dim // se_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(dim // se_ratio, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.flow=='calibrator':
            se_weight = self.se(x)
            out = x * se_weight

        elif self.flow=='adapter':
            mlp_out= self.norm(self.mlp(x))
            return mlp_out
        
        elif self.flow=='tosca_r':
            se_weight = self.se(x)
            mlp_out= self.norm(self.mlp(x))
            out = x * se_weight + mlp_out

        elif self.flow=='tosca':
            mlp_out= self.norm(x + self.mlp(x))
            se_weight = self.se(mlp_out)
            out = mlp_out * se_weight

        return out

class ToscaViT(nn.Module):
    def __init__(self, model, mlp_ratio, se_ratio, flow):
        super().__init__()
        self.vit = model
        self.embed_dim = self.vit.embed_dim

        for param in self.vit.parameters():
            param.requires_grad = False

        self.tosca = Tosca(self.embed_dim, mlp_ratio, se_ratio, flow)

    def forward_features(self, x):
        x = self.vit.forward_features(x)
        x = self.vit.pool(x)
        x = self.vit.fc_norm(x)
        return x

    def forward_tosca(self, x):
        x = self.tosca(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        x = self.forward_tosca(x)
        return x
    
    def reset_tosca(self):
        for m in self.tosca.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight) 
                init.zeros_(m.bias)