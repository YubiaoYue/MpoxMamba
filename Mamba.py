import torch
import torch.nn as nn
from typing import List
from featurefusion import SS2D
from ECAAttention import ECAAttention


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class groupSSM(nn.Module):
    def __init__(self, d_model):
        super(groupSSM, self).__init__()
        self.SSM_1 = SS2D(d_model=d_model//4)
        self.SSM_2 = SS2D(d_model=d_model//4)
        self.SSM_3 = SS2D(d_model=d_model//4)
        self.SSM_4 = SS2D(d_model=d_model//4)

    def forward(self,x):
        x = x.permute(0,2,3,1)
        x1, x2, x3, x4 = x.chunk(4,dim=-1)
        out1 = self.SSM_1(x1)
        out2 = self.SSM_2(x2)
        out3 = self.SSM_3(x3)
        out4 = self.SSM_4(x4)
        out = torch.cat([out1,out2,out3,out4],dim=-1)
        out = out.permute(0,3,1,2)
        return out


class GroupSSMConv(nn.Module):
    def __init__(self,in_c,out_c):
        super(GroupSSMConv, self).__init__()

        self.localrep = nn.Sequential(nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=3,stride=1,groups=in_c,padding=1),
                                      nn.BatchNorm2d(out_c),
                                      nn.SiLU(),
                                      nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=1,stride=1),
                                      nn.BatchNorm2d(out_c),
                                      nn.SiLU())

        self.groupssm = groupSSM(d_model=in_c)
        self.final_conv = nn.Sequential(nn.Conv2d(in_channels=in_c*2,out_channels=out_c,kernel_size=1,stride=1))
        self.eca = ECAAttention(kernel_size=3)
        self.act = nn.SiLU()

    def forward(self,x):
        shortcut = x
        out = self.localrep(x)
        local_temp = self.eca(out)
        out = self.groupssm(out)
        out = self.act(out)
        out = self.final_conv(torch.cat([local_temp,out],dim=1))
        out = self.act(out)
        return out + shortcut

class MpoxMamba(nn.Module):
    def __init__(self,stages_out_channels: List[int],in_c=3,num_classes=10):
        super(MpoxNet, self).__init__()

        self.block1 = nn.Conv2d(in_channels=in_c, out_channels=stages_out_channels[0], kernel_size=3, stride=2, padding=1)
        self.block2 = InvertedResidual(in_channel=stages_out_channels[0],out_channel=stages_out_channels[0],stride=1,expand_ratio=2)
        self.block3 = InvertedResidual(in_channel=stages_out_channels[0],out_channel=stages_out_channels[1],stride=2,expand_ratio=2)
        self.block4 = GroupSSMConv(in_c=stages_out_channels[1],out_c=stages_out_channels[1])
        self.block5 = GroupSSMConv(in_c=stages_out_channels[1],out_c=stages_out_channels[1])
        self.block6 = InvertedResidual(in_channel=stages_out_channels[1],out_channel=stages_out_channels[2],stride=2,expand_ratio=2)
        self.block7 = GroupSSMConv(in_c=stages_out_channels[2],out_c=stages_out_channels[2])
        self.block8 = GroupSSMConv(in_c=stages_out_channels[2],out_c=stages_out_channels[2])
        self.block9 = GroupSSMConv(in_c=stages_out_channels[2],out_c=stages_out_channels[2])
        self.block10 = InvertedResidual(in_channel=stages_out_channels[2],out_channel=stages_out_channels[3],stride=2,expand_ratio=2)
        self.finalconv = nn.Sequential(
            nn.Conv2d(stages_out_channels[3], 512, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True)
        )
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),
                                        nn.Flatten(start_dim=1),
                                        nn.Linear(512,num_classes))
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.finalconv(x)
        x = self.classifier(x)
        return x

# # 假设输入张量形状为 [batch_size, channels, height, width]
input_tensor = torch.randn(1,3,224,224).cuda()
model = MpoxMamba([32,64,128,256]).cuda()
output = model(input_tensor)
# 打印输出张量的形状以验证尺寸
print(output.shape)  # 应该输出 torch.Size([1, 256, 28, 28])
