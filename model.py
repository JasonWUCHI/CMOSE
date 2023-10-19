import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter 
from transformers import VivitModel
from huggingface_hub import hf_hub_download
import torchvision.models as models
from TCN import TemporalConvNet


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
  

class ResnetTCN(nn.Module):
    def __init__(self, n_features, n_hidden, output_dim):
        super(ResnetTCN, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, n_hidden))
        self.n_hidden = n_hidden
        self.channel_sizes = [n_hidden]*8
        self.tcn = TemporalConvNet(
            n_features,
            self.channel_sizes,
            kernel_size=7,
            dropout=0.25
        )
        self.fc1 = nn.Linear(n_hidden, 32)
        self.fc2 = NormedLinear(32,output_dim)


    def forward(self, x_3d):
        x = x_3d.view(-1, 3, 224, 224)

        ax = self.resnet(x)
        ax = ax.view(x_3d.size(0), x_3d.size(1), self.n_hidden)

        # for t in range(x_3d.size(0)):
        #     x = self.resnet(x_3d[t, : , :, :, :]) 
        #     print(x.shape)
        #     print(ax[t].shape) 
        #     print(torch.equal(x, ax[t]))

        ax = ax.transpose(1,2)

        out = self.tcn(ax)       
        out = out[:,:,-1]
        out = out.view(out.size(0), out.size(1))

        x = self.fc1(out)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x, out


