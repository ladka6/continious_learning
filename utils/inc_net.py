import copy
import torch
from torch import nn
from backbone.linears import CosineLinear
import timm

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    type = args["model_name"].lower()
    
    if type == "adaptive_gating_tosca":
        ffn_num = args["ffn_num"]
        from backbone import vit_adapter
        from backbone import adaptive_tosca
        from easydict import EasyDict
        tuning_config = EasyDict(
                    # AdaptFormer
                    ffn_adapt=True,
                    ffn_option="parallel",
                    ffn_adapter_layernorm_option="none",
                    ffn_adapter_init_option="lora",
                    ffn_adapter_scalar="0.1",
                    ffn_num=ffn_num,
                    d_model=768,
                    # VPT related
                    vpt_on=False,
                    vpt_num=0,
                )
        if name == "pretrained_vit_b16_224_adapter" or name == "vit_b16_224_adapter":
            model = vit_adapter.vit_base_patch16_224_adapter(num_classes=0,
                        global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            
            model = adaptive_tosca.ToscaAdaptiveViT(model, args["mlp_ratio"], args["se_ratio"], args['flow'], args['M'])
            model.out_dim=768
            return model.eval()
        elif name == "pretrained_vit_b16_224_in21k_adapter" or name == "vit_b16_224_in21k_adapter":
            model = vit_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                        global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
        
            model = adaptive_tosca.ToscaAdaptiveViT(model, args["mlp_ratio"], args["se_ratio"], args['flow'], args['M'])
            model.out_dim=768
            return model.eval()
        else:
            raise NotImplementedError("Unknown type {}".format(name))
    
    elif name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        if args["model_name"]=="tosca":
            from backbone import vit_tosca
            model= vit_tosca.ToscaViT(model, args["mlp_ratio"], args["se_ratio"], args["flow"])
        model.out_dim = 768
        return model.eval()
    
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        if args["model_name"]=="tosca":
            from backbone import vit_tosca
            model= vit_tosca.ToscaViT(model, args["mlp_ratio"], args["se_ratio"], args["flow"])
        model.out_dim = 768
        return model.eval()

    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out


class SimpleAdaptiveVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    @property
    def feature_dim(self):
        return self.backbone.M

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(
                    nb_classes - self.fc.out_features, feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        return CosineLinear(in_dim, out_dim)

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)         # [B, M]
        out = self.fc(x)             # cosine logits over classes
        out.update({"features": x})
        return out