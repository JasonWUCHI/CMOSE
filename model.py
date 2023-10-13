import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter 
from transformers import VivitModel
from huggingface_hub import hf_hub_download


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class Vivit_backbone(nn.Module):
    def __init__(self, output_dim = 1):
        super().__init__()
        self.vivit_backbone = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.mlp=NormedLinear(768,output_dim)

    def forward(self, x):
        y = self.vivit_backbone(**x)
        y = y.last_hidden_state
        y = torch.mean(y,dim=1)
        score = self.mlp(y)

        return score, y
  




