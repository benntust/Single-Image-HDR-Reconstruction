import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            #nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            #nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0,bias=True)


    def forward(self,x):
        x_3x3 = self.conv(x)
        x_1x1 = self.conv1x1(x)
        return x_3x3+x_1x1

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode = 'bilinear'),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    #nn.BatchNorm2d(ch_out),
			nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            #nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            #nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            #nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class SpatialAttention_block(nn.Module):
    def __init__(self,F_g,F_l):
        super(SpatialAttention_block,self).__init__()
        self.W_g = nn.Conv2d(F_g, 1, kernel_size=1,stride=1,padding=0,bias=True)
        self.W_x = nn.Conv2d(F_l, 1, kernel_size=1,stride=1,padding=0,bias=True)
        self.conv_7x7_g = nn.Conv2d(3, 1, kernel_size=7,stride=1,padding='same',bias=True)
        self.conv_7x7_x = nn.Conv2d(3, 1, kernel_size=7,stride=1,padding='same',bias=True)
        
    def forward(self,g,x):
        avg_xout = torch.mean(x, dim=1, keepdim=True)
        max_xout, _ = torch.max(x, dim=1, keepdim=True)
        conv_xout = self.W_x(x)
        xout = torch.cat([avg_xout, max_xout, conv_xout], dim=1)
        avg_gout = torch.mean(g, dim=1, keepdim=True)
        max_gout, _ = torch.max(g, dim=1, keepdim=True)
        conv_gout = self.W_g(g)
        gout = torch.cat([avg_gout, max_gout, conv_gout], dim=1)
        x1 = self.conv_7x7_x(xout)
        g1 = self.conv_7x7_g(gout)
        out = torch.sigmoid(x1+g1)

        return out*x

class ChannelAttention_block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention_block,self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_conv_en_x = nn.Conv2d(channel, mid_channel, kernel_size=1,stride=1,padding=0,bias=True)
        self.shared_conv_en_g = nn.Conv2d(channel, mid_channel, kernel_size=1,stride=1,padding=0,bias=True)
        self.shared_conv_de = nn.Conv2d(mid_channel, channel, kernel_size=1,stride=1,padding=0,bias=True)
        
    def forward(self,g,x):
        avg_xout = self.avg_pool(x)
        max_xout = self.max_pool(x)
        avg_xout = self.shared_conv_en_x(avg_xout)
        max_xout = self.shared_conv_en_x(max_xout)
        avg_gout = self.avg_pool(g)
        max_gout = self.max_pool(g)
        avg_gout = self.shared_conv_en_g(avg_gout)
        max_gout = self.shared_conv_en_g(max_gout)
        out = avg_xout + max_xout + avg_gout + max_gout
        out = self.shared_conv_de(out)
        out = torch.sigmoid(out)

        return out*x

class Sam_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=3):
        super(Sam_Net,self).__init__()
        
        self.Avgpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1_2 = conv_block(ch_in=img_ch*3,ch_out=64)
        self.Conv2_2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3_2 = conv_block(ch_in=128,ch_out=256)
        self.Conv4_2 = conv_block(ch_in=256,ch_out=512)

        self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        self.Conv6 = conv_block(ch_in=1024,ch_out=1024)
        self.Conv7 = conv_block(ch_in=1024,ch_out=1024)
        #self.Conv8 = conv_block(ch_in=512,ch_out=512)
        #self.Conv9 = conv_block(ch_in=512,ch_out=512)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.SpatialAtt5_n = SpatialAttention_block(F_g=512,F_l=512)
        self.ChannelAtt5_n = ChannelAttention_block(channel=512)
        self.Up_conv5 = conv_block(ch_in=512*2, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.SpatialAtt4_n = SpatialAttention_block(F_g=256,F_l=256)
        self.ChannelAtt4_n = ChannelAttention_block(channel=256)
        self.Up_conv4 = conv_block(ch_in=256*2, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.SpatialAtt3_n = SpatialAttention_block(F_g=128,F_l=128)
        self.ChannelAtt3_n = ChannelAttention_block(channel=128)
        self.Up_conv3 = conv_block(ch_in=128*2, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.SpatialAtt2_n = SpatialAttention_block(F_g=64,F_l=64)
        self.ChannelAtt2_n = ChannelAttention_block(channel=64)
        self.Up_conv2 = conv_block(ch_in=64*2, ch_out=64)

        self.Conv_1x1 = nn.Sequential(
                            nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0),
                            nn.ReLU(inplace=True)
                        )


    def forward(self,x1,x2,x3):
        # encoding path
        x = torch.cat((x1,x2,x3),dim=1)

        x2_1 = self.Conv1_2(x)
        x2_2 = self.Avgpool(x2_1)
        x2_2 = self.Conv2_2(x2_2)
        x2_3 = self.Avgpool(x2_2)
        x2_3 = self.Conv3_2(x2_3)
        x2_4 = self.Avgpool(x2_3)
        x2_4 = self.Conv4_2(x2_4)
        x2_ = self.Avgpool(x2_4)

        x5 = self.Conv5(x2_)
        x5 = self.Conv6(x5)
        x5 = self.Conv7(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4_n = self.SpatialAtt5_n(g=d5,x=x2_4)
        x4_n = self.ChannelAtt5_n(g=d5,x=x4_n)
        d5 = torch.cat((x4_n,d5),dim=1)      
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3_n = self.SpatialAtt4_n(g=d4,x=x2_3)
        x3_n = self.ChannelAtt4_n(g=d4,x=x3_n)
        d4 = torch.cat((x3_n,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2_n = self.SpatialAtt3_n(g=d3,x=x2_2)
        x2_n = self.ChannelAtt3_n(g=d3,x=x2_n)
        d3 = torch.cat((x2_n,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1_n = self.SpatialAtt2_n(g=d2,x=x2_1)
        x1_n = self.ChannelAtt2_n(g=d2,x=x1_n)
        d2 = torch.cat((x1_n,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        #out = x + d1

        return d1


if __name__ == '__main__':
    from torchinfo import summary
    model = Sam_Net()
    summary(model, input_size=((1,3,512,512),(1,3,512,512),(1,3,512,512)))