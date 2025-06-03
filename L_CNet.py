import warnings

from module.Att.CPCA import CPCABlock
from module.Att.MSCA import MultiSpectralAttentionLayer
from module.Conv.DOConv import DOConv2d

warnings.filterwarnings("ignore")
from module.Att.HILO_ATT.hilo import HiLo
from module.Conv.HWD.hwd import HWDownsampling
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Flatten
from thop import profile

from module.Att import EMA, ELA, TripletAttention, SCSA,CPCA
from module.Conv import Partial_conv3, SPDConv, ScConv, RFAConv, WTConv2d, LDConv, SRU, CRU


# --------------------------------------------------------

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)     # 3 1
        # self.conv = DOConv2d(in_channels,out_channels,3,padding=1)
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)  # 3 1
        # self.conv = WTConv2d(in_channels, out_channels, kernel_size=3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class CBN_3_1(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN_3_1, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)  # 1 0
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        # self.conv = nn.Conv2d(in_channels, in_channels,kernel_size=1, padding=0)     # 1 0
        # self.conv = WTConv2d(in_channels, out_channels, kernel_size=3)

        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.norm(out)
        return self.activation(out)


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, groups=groups)
        # self.depthwise.apply(weights_init)
        # self.pointwise.apply(weights_init)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# CCA attention
class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class LCL(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(LCL, self).__init__()
        self.layer1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
                                   stride=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ECA(out_channels),
        )
        # self.layer1.apply(weights_init)

    def forward(self, x):
        out = self.layer1(x)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv,activation='ReLU'):
        super(UpBlock, self).__init__()
        # 单只上采样
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2, padding=0, bias=False)
        # self.up = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)
        # 双支并行上采样
        # self.up_1 = nn.Upsample(scale_factor=2, mode='bilinear')    # 默认最近另邻插值，bilinear是双线性插值
        # self.up_2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2, padding=0,
        #                                bias=False)
        # self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0)  # 3 1
        # self.norm = nn.BatchNorm2d(in_channels // 2)
        # self.activation = get_activation(activation)

        # self.up = DySample(in_channels=in_channels // 2)
        self.coatt = CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        # self.coatt = TripletAttention()
        # self.coatt = CGAFusion(dim=in_channels // 2)
        # self.coatt_branch = ECA(out_channels // 2)
        # self.coatt_branch = EMA(channels=in_channels, factor=in_channels // 2)

        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        # self.nConvs = Res_block(in_channels, out_channels)
        # self.conv_1 = CBN(in_channels, out_channels, activation)
        # self.conv_2 = CBN_1x1(out_channels, out_channels, activation)
        # self.nConvs = CBN_3_1(in_channels, out_channels, activation)

    def forward(self, x, skip_x):
        # print('x=', x.size())
        # print('skip_x=', skip_x.size())
        # res = skip_x

        # 单只上采样
        up = self.up(x)

        # print(up.size())
        # 双支并行上采样
        # m = self.up_1(x)
        # b = self.up_2(x)
        # cat = torch.cat([m, b], dim=1)
        # up = self.conv(cat)
        # up = self.norm(up)
        # up = self.activation(up)

        skip_x_att = self.coatt(g=up, x=skip_x)
        # up = self.branch(up)
        # skip_x = self.branch(skip_x)
        # skip_x_att_up = self.coatt(up) # CCA
        # skip_x_att_skip_x = self.coatt(skip_x) # CCA
        # skip_x_att = self.coatt(up) # ECA
        # skip_x_att = self.coatt(up)   # triplet
        # skip_x_att = self.coatt(up,skip_x)   # CGAfusion
        # if skip_x_att.size(2) != up.size(2) or skip_x_att.size(3) != up.size(3):
        #     up = F.interpolate(up, size=(skip_x_att.size(2), skip_x_att.size(3)), mode='bilinear', align_corners=True)
        # print('skip_x_att=', skip_x_att.size())
        # print('skip_x_att_up=',skip_x_att_up.size())
        # print('skip_x_att_skip_x=',skip_x_att_skip_x.size())
        # x = torch.cat([skip_x, up], dim=1)  # dim 1 is the channel dimension    256 32 32
        # x = self.coatt_branch(x)
        # x = x * res
        # print('x = ', x.size())
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        # print('x=', x.size())
        # x = self.branch(x)
        # x = self.conv_1(x)
        # x = self.conv_2(x)
        # print(x.size())
        return self.nConvs(x)



class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.branch_1 = nn.MaxPool2d(2, 2)
        self.branch_2 = SPDConv(in_channels, in_channels)

        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)
        # self.coatt = ECA(out_channels // 2)
        # self.coatt = EMA(channels=in_channels // 2,factor=in_channels // 4)

    def forward(self, x):
        d_1 = self.branch_1(x)
        d_2 = self.branch_2(x)
        cat = torch.cat([d_1, d_2], dim=1)
        r = self.conv(cat)
        r = self.norm(r)
        r = self.activation(r)
        return r


class Skip_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Skip_block, self).__init__()
        # self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv1 = Res_block(in_channels, out_channels)
        self.skip = TripletAttention()
        # self.skip = ECA(out_channels)
        # self.skip = EMA(channels=in_channels, factor=in_channels // 2)

    def forward(self, x):
        # x = self.conv1(x)
        x = self.skip(x)
        return x


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.01):
        super(Res_block, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)  # 3 1
        # self.conv1 = ScConv(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 自改Conv -----------------------------------------------------------------------------
        # self.conv2 = Partial_conv3(dim=out_channels,n_div=2,forward='split_cat')    # 3.7900G 1.905533M FPS 81
        # self.conv2 = DOConv2d(out_channels, out_channels, 3, padding=1)
        self.conv2 = WTConv2d(out_channels, out_channels, kernel_size=3)  # # 3.372G 1.684637M FPS 62
        # self.conv2 = RFAConv(out_channels, out_channels, kernel_size=3)  # 输入输出均可改，卷积核可改 5.9048G 2.88833M FPS 57
        # self.conv2 = ScConv(out_channels)                                           # 3.7017G 1.852285M FPS 33
        # self.conv2 = DepthwiseSeparableConv(out_channels,out_channels,3,padding=1)          # 90
        # --------------------------------------------------------------------------------------
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        # self.ema = EMA(out_channels,out_channels // 2)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        # ------------------------------------------
        # self.SRU = SRU(out_channels,group_num=4,gate_treshold=0.5)
        # self.CRU = CRU(out_channels,alpha=1 / 2,squeeze_radio=2,group_size=2,group_kernel_size=3)
        # self.ca = ChannelAttention(in_planes=out_channels,ratio=8)
        # self.sa = SpatialAttention(kernel_size=3)
        # self.cpca = CPCABlock(out_channels, out_channels,channelAttention_reduce=4)
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.conv3(out)
        # temp = out
        # out = self.ca(out) * out
        # print(out.size())
        # out = self.sa(out) * out
        # out = self.cpca(out)
        # out = self.ema(out)
        # out = self.SRU(out)
        #
        # out = self.CRU(out)

        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fusion, self).__init__()
        self.conv1 = RFAConv(in_channels, out_channels, kernel_size=1)
        # self.conv1 = LDConv(in_channels, out_channels, 2)
        self.conv2 = WTConv2d(in_channels, out_channels, kernel_size=3)
        # self.conv2 = ScConv(out_channels)

    def forward(self, x):
        residual = x
        residual = self.conv1(residual)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv1(out)
        out += residual
        return out


class LC_Net(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, img_size=256, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 8  # basic channel 64
        block = Res_block
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)

        # self.ela_1 = ELA(in_channels, phi='B')
        # self.ela_2 = ELA(in_channels * 2, phi='B')
        # self.ela_3 = ELA(in_channels * 4, phi='B')
        # self.ela_4 = ELA(in_channels * 8, phi='B')
        # self.ela_5 = ELA(in_channels * 8, phi='B')

        self.ema_1 = EMA(in_channels, in_channels // 2)
        self.ema_2 = EMA(in_channels * 2, (in_channels * 2) // 2)
        self.ema_3 = EMA(in_channels * 4, (in_channels * 4) // 2)
        self.ema_4 = EMA(in_channels * 8, (in_channels * 8) // 2)
        self.ema_5 = EMA(in_channels * 8, (in_channels * 8) // 2)


        # self.spd_tri_1 = spd_tri(in_channels=1,out_channels=in_channels)
        # self.spd_tri_2 = spd_tri(in_channels=in_channels,out_channels=in_channels * 2)
        # self.spd_tri_3 = spd_tri(in_channels=in_channels * 2,out_channels= in_channels * 4)
        # self.spd_tri_4 = spd_tri(in_channels=in_channels * 4,out_channels= in_channels * 8)
        # self.spd_tri_5 = spd_tri(in_channels=in_channels * 8,out_channels= in_channels * 8)

        # # SPDConv 通道数可调，尺寸减半; 尝试替代MaxPool2d池化层
        self.SPDConv_1 = SPDConv(c1=in_channels, c2=in_channels)
        self.SPDConv_2 = SPDConv(c1=in_channels * 2, c2=in_channels * 2)
        self.SPDConv_3 = SPDConv(c1=in_channels * 4, c2=in_channels * 4)
        self.SPDConv_4 = SPDConv(c1=in_channels * 8, c2=in_channels * 8)

        self.encoder1 = self._make_layer(block, in_channels, in_channels * 2, 2)  # 64  128
        self.encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 2)  # 64  128
        self.encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 2)  # 64  128
        self.encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)  # 64  128
        # self.Fusion_4 = Fusion(128, 128)
        #
        # self.encoder3_1 = self._make_layer(block, 48, 32, 1)  # 64  128
        # self.encoder3_2 = self._make_layer(block, 16, 16, 1)  # 64  128
        # self.encoder3_3 = self._make_layer(block, 32, 32, 1)  # 64  128
        #
        # self.encoder4_1 = self._make_layer(block, 112, 64, 1)  # 64  128
        # self.encoder4_2 = self._make_layer(block, 16, 16, 1)  # 64  128
        # self.encoder4_3 = self._make_layer(block, 32, 32, 1)  # 64  128
        # self.encoder4_4 = self._make_layer(block, 64, 64, 1)  # 64  128
        #
        # self.encoder5_1 = self._make_layer(block, 240, 128, 1)  # 64  128
        # self.encoder5_2 = self._make_layer(block, 16, 16, 1)  # 64  128
        # self.encoder5_3 = self._make_layer(block, 32, 32, 1)  # 64  128
        # self.encoder5_4 = self._make_layer(block, 64, 64, 1)  # 64  128
        # self.encoder5_5 = self._make_layer(block, 128, 128, 1)  # 64  128

        # 编码器顺序连接
        # self.stage_1 = self._make_layer(block, in_channels * 3, in_channels * 2, 3)
        # self.stage_2 = self._make_layer(block, in_channels * 6, in_channels * 4, 3)
        # self.stage_3 = self._make_layer(block, in_channels * 12, in_channels * 8, 3)
        # self.stage_4 = self._make_layer(block, in_channels * 16, in_channels * 8, 3)
        # 16
        # self.stage_1 = DepthwiseSeparableConv(in_channels=48, out_channels=32, kernel_size=1, padding=0)
        # self.stage_2 = DepthwiseSeparableConv(in_channels=96, out_channels=64, kernel_size=1, padding=0)
        # self.stage_3 = DepthwiseSeparableConv(in_channels=192, out_channels=128, kernel_size=1, padding=0)
        # self.stage_4 = DepthwiseSeparableConv(in_channels=256, out_channels=128, kernel_size=1, padding=0)
        # self.stage_1 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, padding=0)
        # self.stage_2 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1, padding=0)
        # self.stage_3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, padding=0)
        # self.stage_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0)
        # 8
        # self.stage_1 = DepthwiseSeparableConv(in_channels=48, out_channels=32, kernel_size=1, padding=0)
        # self.stage_2 = DepthwiseSeparableConv(in_channels=96, out_channels=64, kernel_size=1, padding=0)
        # self.stage_3 = DepthwiseSeparableConv(in_channels=192, out_channels=128, kernel_size=1, padding=0)
        # self.stage_4 = DepthwiseSeparableConv(in_channels=256, out_channels=128, kernel_size=1, padding=0)
        # self.stage_1 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=1, padding=0)
        # self.stage_2 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, padding=0)
        # self.stage_3 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1, padding=0)
        # self.stage_4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0)
        # decode_stage
        # self.stage_5 = self._make_layer(block, 192, 64, 1)
        # self.stage_6 = self._make_layer(block, 96, 32, 1)
        # self.stage_7 = self._make_layer(block, 48, 16, 1)
        # self.stage_8 = self._make_layer(block, 32, 16, 1)
        # self.stage_9 = self._make_layer(block, 240, 128, 1)
        # self.up = nn.Upsample(scale_factor=2)
        # self.up_stage_3 = UpBlock(96, 32, nb_Conv=2)

        # self.mtc_1 = LCL(in_channels=in_channels, out_channels=in_channels)
        # self.mtc_2 = LCL(in_channels=in_channels * 2, out_channels=in_channels * 2)
        # self.mtc_3 = LCL(in_channels=in_channels * 4, out_channels=in_channels * 4)
        # self.mtc_4 = LCL(in_channels=in_channels * 8, out_channels=in_channels * 8)

        # TripletAttention 不改变图像的通道数和尺寸
        # self.RFAConv_1 = RFAConv(in_channels, in_channels, kernel_size=3)
        # self.RFAConv_2 = RFAConv(in_channels * 2, in_channels * 2, kernel_size=3)
        # self.RFAConv_3 = RFAConv(in_channels * 4, in_channels * 4, kernel_size=3)
        # self.RFAConv_4 = RFAConv(in_channels * 8, in_channels * 8, kernel_size=3)
        # self.mtc_1 = TripletAttention()
        # self.mtc_2 = TripletAttention()
        # self.mtc_3 = TripletAttention()
        # self.mtc_4 = TripletAttention()
        # self.mtc_1 = Skip_block(in_channels, in_channels)
        # self.mtc_2 = Skip_block(in_channels * 2, in_channels * 2)
        # self.mtc_3 = Skip_block(in_channels * 4, in_channels * 4)
        # self.mtc_4 = Skip_block(in_channels * 8, in_channels * 8)
        self.mtc_1 = MultiSpectralAttentionLayer(in_channels,256,256,8,'top8')
        self.mtc_2 = MultiSpectralAttentionLayer(in_channels * 2,128,128,8,'top8')
        self.mtc_3 = MultiSpectralAttentionLayer(in_channels * 4,64,64,8,'top8')
        self.mtc_4 = MultiSpectralAttentionLayer(in_channels * 8,32,32,8,'top8')
        self.conv_1 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_2 = DepthwiseSeparableConv(in_channels * 2, in_channels * 2, kernel_size=3, padding=1)
        self.conv_3 = DepthwiseSeparableConv(in_channels * 4, in_channels * 4, kernel_size=3, padding=1)
        self.conv_4 = DepthwiseSeparableConv(in_channels * 8, in_channels * 8, kernel_size=3, padding=1)
        # self.mtc_1 = EMA(16,8)
        # self.mtc_2 = EMA(32,16)
        # self.mtc_3 = EMA(64,32)
        # self.mtc_4 = EMA(128,64)
        # # self.mtc_5 = TripletAttention()
        # self.mtc_1 = SCSA(in_channels,2)
        # self.mtc_2 = SCSA(in_channels * 2,2)
        # self.mtc_3 = SCSA(in_channels * 4,2)
        # self.mtc_4 = SCSA(in_channels * 8,2)

        # self.SCConv_1 = ScConv(in_channels)
        # self.SCConv_2 = ScConv(in_channels * 2)
        # self.SCConv_3 = ScConv(in_channels * 4)
        # self.SCConv_4 = ScConv(in_channels * 8)
        # self.SCConv_5 = ScConv(in_channels * 8)

        # 小波下采样

        self.maxpooling_16 = nn.MaxPool2d(kernel_size=(16, 16), stride=(16, 16))
        self.maxpooling_8 = nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8))
        self.maxpooling_4 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.maxpooling_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


        self.decoder4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.decoder3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.decoder2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)
        self.decoder1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)
        # self.decoder4 = UpBlock(in_channels * 16, in_channels * 4)
        # self.decoder3 = UpBlock(in_channels * 8, in_channels * 2)
        # self.decoder2 = UpBlock(in_channels * 4, in_channels)
        # self.decoder1 = UpBlock(in_channels * 2, in_channels)



        # self.conv_1 = nn.Conv2d(64, 64, kernel_size=1)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print('model_input = ',x.shape)  # test 1 1 512 512
        x1 = self.inc(x)  # 16 256 256
        # x1 = self.ela_1(x1)
        x1 = self.ema_1(x1)
        # x1 = self.MSCA_1(x1)
        # x1 = self.SCConv_1(x1)
        # x2 = self.encoder1(self.pool(x1))
        # # x2 = self.SCConv_2(x2)
        # # # x2 = self.ela_1(x2)
        # #
        # x3 = self.encoder2(self.pool(x2))
        # # x3 = self.SCConv_3(x3)
        # # # x3 = self.ela_2(x3)
        # #
        # x4 = self.encoder3(self.pool(x3))
        # # x4 = self.SCConv_4(x4)
        # # # x4 = self.ela_3(x4)
        #
        # # d5 = self.SCConv_5(d5)
        # # d5 = self.ela_4(d5)

        # SPDConv   替换pool  并行采样
        # x2 = self.encoder1(self.parallel_down_1(x1))  # 32 128 128
        # print(x2.size())
        x2 = self.encoder1(self.SPDConv_1(x1))  # 32 128 128
        # x2 = self.encoder1(self.hwd_1(x1))  # 32 128 128
        # x2 = self.ela_2(x2)
        x2 = self.ema_2(x2)
        # x2 = self.MSCA_2(x2)
        # x2 = self.SCConv_2(x2)
        # x2 = self.stage_1(torch.cat((self.maxpooling_2(x1), x2), dim=1))  # 32 128 128

        # a = self.encoder3_2(self.maxpooling_2(x1))
        # b = self.encoder3_3(x2)

        # x3 = self.encoder3_1(torch.cat((self.encoder3_2(self.maxpooling_2(x1)),
        #                                     self.encoder3_3(x2)), dim=1))
        # print(x3.size())    # 32 128 128
        # x3 = self.encoder2(self.parallel_down_2(x2))  # 64 64 64
        x3 = self.encoder2(self.SPDConv_2(x2))  # 64 64 64
        # x3 = self.encoder2(self.hwd_2(x2))
        # x3 = self.ela_3(x3)
        # x3 = self.MSCA_3(x3)
        x3 = self.ema_3(x3)
        # x3 = self.SCConv_3(x3)
        # x3 = self.stage_2(torch.cat((self.maxpooling_2(x2), x3), dim=1))  # 64 64 64

        # x4 = self.encoder4_1(torch.cat((self.maxpooling_4(x1),
        #                                         self.encoder4_3(self.maxpooling_2(x2)),
        #                                         self.encoder4_4(x3)), dim=1))  #
        # print(x4.size())    # 64 64 64
        # x4 = self.encoder3(self.parallel_down_3(x3))  # 128 32 32
        x4 = self.encoder3(self.SPDConv_3(x3))  # 128 32 32
        # x4 = self.encoder3(self.hwd_3(x3))
        # x4 = self.ela_4(x4)
        x4 = self.ema_4(x4)
        # x4 = self.MSCA_4(x4)
        # x4 = self.SCConv_4(x4)
        # x4 = self.stage_3(torch.cat((self.maxpooling_2(x3), x4), dim=1)) # 128 32 32

        # x5 = self.encoder5_1(torch.cat((self.encoder5_2(self.maxpooling_8(x1)),
        #                                         self.encoder5_3(self.maxpooling_4(x2)),
        #                                         self.encoder5_4(self.maxpooling_2(x3)),
        #                                         self.encoder5_5(x4)),dim=1)) #
        # print(x5.size())    # 128 32 32
        # 聚合Fusion_4
        # d5 = self.encoder4(self.parallel_down_4(x4))  # 128 16 16
        d5 = self.encoder4(self.SPDConv_4(x4))  # 128 16 16
        # d5 = self.encoder4(self.hwd_4(x4))
        # d5 = self.Fusion_4(self.SPDConv_4(x5))  # 128 16 16
        # d5 = self.ela_5(d5)
        # d5 = self.MSCA_5(d5)
        d5 = self.ema_5(d5)
        # d5 = self.SCConv_5(d5)
        # d5 = self.stage_4(torch.cat((self.maxpooling_2(x4), d5), dim=1))  # 128 16 16
        # d5 = self.conv_1(d5)

        # x1 = self.spd_tri_1(x)  # 16 128 128
        # x2 = self.spd_tri_2(x1) # 32 64 64
        # x3 = self.spd_tri_3(x2) # 64 32 32
        # # print(x3.size())
        # x4 = self.spd_tri_4(x3) # 128 16 16
        # print('x4=',x4.size())
        # # d5 = self.spd_tri_5(x4) # 1
        # # d5 = self.mtc_5(x4)
        # d5 = self.encoder4(x4)
        # print('d5=',d5.size())

        f1 = x1
        f2 = x2
        f3 = x3
        f4 = x4
        # f1 = torch.cat((self.maxpooling_2(x1), x2), dim=1)  #
        # f2 = self.stage_1(torch.cat((self.maxpooling_2(x1), x2), dim=1))  #
        # f3 = self.stage_2(torch.cat((self.maxpooling_2(x2), x3), dim=1))  #
        # f4 = self.stage_3(torch.cat((self.maxpooling_2(x3), x4), dim=1))  #
        x1 = self.conv_1(f1)
        x2 = self.conv_2(f2)
        x3 = self.conv_3(f3)
        x4 = self.conv_4(f4)
        x1 = self.mtc_1(x1)  # 16 256 256
        x2 = self.mtc_2(x2)  # 32 128 128
        x3 = self.mtc_3(x3)  # 64 64 64
        x4 = self.mtc_4(x4)  # 128 32 32
        x1 = f1 + x1
        x2 = f2 + x2
        x3 = f3 + x3
        x4 = f4 + x4
        # d5 = F.interpolate(d5, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.decoder4(d5, x4)  # 64 32 32
        # d4 = self.stage_5(torch.cat((self.up(d5), d4), dim=1))  # 64 32 32
        # print(d4.size())

        d3 = self.decoder3(d4, x3)  # 32 64 64
        # d3 = self.up_stage_3(d4, x3)  # 32 64 64
        # d3 = self.stage_6(torch.cat((self.up(d4), d3), dim=1))  # 32 64 64
        # print(d3.size())
        d2 = self.decoder2(d3, x2)  # 16 128 128
        # d2 = self.stage_7(torch.cat((self.up(d3), d2), dim=1))  # 16 128 128
        # print('d2=',d2.size())
        d1 = self.decoder1(d2, x1)  # 16 256 256
        # d1 = self.stage_8(torch.cat((self.up(d2), d1), dim=1))  # 16 256 256
        # print('d4=',d4.size())
        # print('d3=',d3.size())
        # print('d2=',d2.size())
        # print('d1=',d1.size())
        out = self.outc(d1)
        return out.sigmoid()

#
# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         nn.init.constant_(m.bias, 0)
#     return m


if __name__ == '__main__':
    import time
    import torch

    model = LC_Net()
    # model = ChannelAttention(in_planes=8,ratio=8)
    # model = SpatialAttention(kernel_size=3)
    # model = DepthwiseSeparableConv(48, 32, 1)
    # print(model)
    inputs = torch.rand(1, 1, 256, 256)

    output = model(inputs)

    flops, params = profile(model, (inputs,))
    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
    print(output.size())

    # 打印参数量
    # 1
    # 遍历每一层，打印参数数量和层的名称
    # def print_param_count(model):
    #     total_params = 0
    #     for name, param in model.named_parameters():
    #         param_count = param.numel()  # 获取该参数的元素数量
    #         total_params += param_count
    #         print(f"Layer: {name}, Parameters: {param_count}")
    #     print(f"Total parameters: {total_params}")
    # print_param_count(model)

    # 2
    # 创建一个随机输入张量，形状与模型输入相匹配
    # input_tensor = torch.randn(1, 1, 256, 256)

    # 定义函数，计算在前向传播过程中使用的模块的参数量
    # def print_forward_modules_param_count(model):
    #     module_params = []
    #     for idx, (name, module) in enumerate(model.named_modules()):
    #         # 只计算包含参数的层（排除激活层、池化层等没有参数的模块）
    #         if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.Conv1d, nn.Conv3d)):
    #             # 获取该模块的所有参数数量
    #             param_count = sum(p.numel() for p in module.parameters())
    #             if param_count > 0:  # 只输出有参数的模块
    #                 module_params.append((f"Layer {idx + 1}: {name}", param_count))
    #     # 按顺序打印每个模块的参数量
    #     for name, param_count in module_params:
    #         print(f"{name} - Parameters: {param_count}")
    #
    #
    # # 调用函数
    # print_forward_modules_param_count(model)

    # 计算FPS
    # 将模型移到 GPU (如果有)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = torch.randn(1, 1, 256, 256).to(device)
    #
    #
    # # 计算1秒内的FPS，忽略第一张输入数据的时间
    def measure_fps_exclude_first_1sec(model, inputs):
        model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 关闭梯度计算，加速推理
            # 第一次推理，不计入时间
            model(inputs)
            # 记录从第二次推理开始的时间
            start_time = time.time()
            num_iters = 0  # 计数模型运行的次数
            while True:
                # 运行模型推理
                output = model(inputs)
                num_iters += 1
                # 检查是否已经过了1秒
                if time.time() - start_time >= 1.0:
                    break
        # 1秒内模型运行的次数就是FPS
        return num_iters


    # 测量FPS
    fps = measure_fps_exclude_first_1sec(model, inputs)
    print(f"Model FPS (excluding first iteration, 1 second): {fps:.2f}")
