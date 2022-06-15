import torchvision.models as models
import torch
from models.mlp_head import MLPHead
from torch.nn import Linear, ReLU, LeakyReLU


class ResNet18(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()
        if kwargs['name'] == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif kwargs['name'] == 'resnet50':
            resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)

class ResNet18Transfer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18Transfer, self).__init__()
        if kwargs['name'] == 'resnet18Transfer':
            self.resnet = models.resnet18(pretrained=True)
        elif kwargs['name'] == 'resnet50Transfer':
            self.resnet = models.resnet50(pretrained=True)

        #self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        # self.encoder[0].weight.requires_grad
        
        # Freeze encoder weights update
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        #for param in self.encoder.parameters():
        #    param.requires_grad = False
        
        #self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])
        # Add new last layer for transfer training. 
        #self.transfer = Linear(resnet.fc.in_features, out_features=2)
        
        self.resnet.fc = Linear(self.resnet.fc.in_features, out_features=2)
        #self.resnet.relu = LeakyReLU(self.resnet.fc)
        #self.relu = ReLU()
        
        #self.transfer['fc.bias'].requires_grad = True
        #self.transfer['fc.weight'].requires_grad = True
        #self.transfer.bias.requires_grad = True
        #self.transfer.weight.requires_grad = True
        self.resnet.fc.bias.requires_grad = True
        self.resnet.fc.weight.requires_grad = True

    def forward(self, x):
        #h = self.encoder(x)
        h = self.resnet(x)
        try:
            if self.relu:
                h = self.relu(h)
        except:
            pass
        h = h.view(h.shape[0], h.shape[1])
        return h
