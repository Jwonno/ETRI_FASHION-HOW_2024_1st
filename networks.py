'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2024.04.20.
'''
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

class ResExtractor(nn.Module):
    """Feature extractor based on ResNet structure
        Selectable from resnet18 to resnet152

    Args:
        resnetnum: Desired resnet version
                    (choices=['18','34','50','101','152'])
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """

    def __init__(self, resnetnum='50', pretrained=True):
        super(ResExtractor, self).__init__()

        if resnetnum == '18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnetnum == '34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnetnum == '50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnetnum == '101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnetnum == '152':
            self.resnet = models.resnet152(pretrained=pretrained)

        self.modules_front = list(self.resnet.children())[:-2]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)


class MnExtractor(nn.Module):
    """Feature extractor based on MobileNetv2 structure
    Args:
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """

    def __init__(self, pretrained=True):
        super(MnExtractor, self).__init__()

        self.net = models.mobilenet_v2(pretrained=pretrained)
        self.modules_front = list(self.net.children())[:-1]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)


class Baseline_ResNet_emo(nn.Module):
    """ Classification network of emotion categories based on ResNet18 structure. """
    
    def __init__(self):
        super(Baseline_ResNet_emo, self).__init__()

        self.encoder = ResExtractor('152')
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.daily_linear = nn.Linear(512, 6)
        self.gender_linear = nn.Linear(512, 5)
        self.embel_linear = nn.Linear(512, 3)

    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.encoder.front(x['image'])
        flatten = self.avg_pool(feat).squeeze()

        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel


class Baseline_MNet_emo(nn.Module):
    """ Classification network of emotion categories based on MobileNetv2 structure. """
    
    def __init__(self):
        super(Baseline_MNet_emo, self).__init__()

        self.encoder = MnExtractor()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.daily_linear = nn.Linear(1280, 6)
        self.gender_linear = nn.Linear(1280, 5)
        self.embel_linear = nn.Linear(1280, 3)

    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.encoder.front(x['image'])
        flatten = self.avg_pool(feat).squeeze()

        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel

    
from tiny_vit import tiny_vit_5m_224
    
class TinyViT(nn.Module):
    def __init__(self, distillation, pretrained=False):
        super(TinyViT, self).__init__()
        
        self.base_model = tiny_vit_5m_224(pretrained=pretrained, num_classes=0, pretrained_type='22k_distill')
        
        self.distillation = distillation
     
        self.embed_dim = 320
        
        self.head_daily = nn.Linear(self.embed_dim, 6)
        self.head_gender = nn.Linear(self.embed_dim, 5)
        self.head_embel = nn.Linear(self.embed_dim, 3)
        
        if self.distillation:
            self.dist_head_daily = nn.Linear(self.embed_dim, 6)
            self.dist_head_gender = nn.Linear(self.embed_dim, 5)
            self.dist_head_embel = nn.Linear(self.embed_dim, 3)

    def forward(self, x):

        feat = self.base_model(x['image'])
        
        if self.distillation:
            out_daily = self.head_daily(feat), self.dist_head_daily(feat)
            out_gender = self.head_gender(feat), self.dist_head_gender(feat)
            out_embel = self.head_embel(feat), self.dist_head_embel(feat)
            
            if not self.training:
                out_daily = (out_daily[0] + out_daily[1]) / 2
                out_gender = (out_gender[0] + out_gender[1]) / 2
                out_embel = (out_embel[0] + out_embel[1]) / 2
        else:
            out_daily = self.head_daily(feat)
            out_gender = self.head_gender(feat)
            out_embel = self.head_embel(feat)
            
        return out_daily, out_gender, out_embel


class MLPClassifier(nn.Module):
    def __init__(self, embed_dim, expanded_dim, num_classes):
        super(MLPClassifier, self).__init__()
        
        self.in_dim = embed_dim
        self.hidden_dim = expanded_dim
        self.out_dim = num_classes
        
        self.feature_expansion = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.01)
        )
        
        self.head = nn.Linear(self.hidden_dim, self.out_dim)
        
    def forward(self, x):
        feat = self.feature_expansion(x)
        out = self.head(feat)
        return out
        

    
class TinyViT_MLP(nn.Module):
    def __init__(self, distillation, pretrained=False):
        super(TinyViT_MLP, self).__init__()
        
        self.base_model = tiny_vit_5m_224(pretrained=pretrained, pretrained_type='22k_distill', num_classes=0)
        
        self.distillation = distillation
     
        self.embed_dim = 320
        self.expanded_dim = 512
        
        
        self.head_daily = MLPClassifier(self.embed_dim, self.expanded_dim, 6)
        self.head_gender = MLPClassifier(self.embed_dim, self.expanded_dim, 5)
        self.head_embel = MLPClassifier(self.embed_dim, self.expanded_dim, 3)
        
        if self.distillation:
            self.dist_head_daily = MLPClassifier(self.embed_dim, self.expanded_dim, 6)
            self.dist_head_gender = MLPClassifier(self.embed_dim, self.expanded_dim, 5)
            self.dist_head_embel = MLPClassifier(self.embed_dim, self.expanded_dim, 3)

    def forward(self, x):

        feat = self.base_model(x['image'])
        
        if self.distillation:
            out_daily = self.head_daily(feat), self.dist_head_daily(feat)
            out_gender = self.head_gender(feat), self.dist_head_gender(feat)
            out_embel = self.head_embel(feat), self.dist_head_embel(feat)
            
            if not self.training:
                out_daily = (out_daily[0] + out_daily[1]) / 2
                out_gender = (out_gender[0] + out_gender[1]) / 2
                out_embel = (out_embel[0] + out_embel[1]) / 2
        else:
            out_daily = self.head_daily(feat)
            out_gender = self.head_gender(feat)
            out_embel = self.head_embel(feat)
            
        return out_daily, out_gender, out_embel