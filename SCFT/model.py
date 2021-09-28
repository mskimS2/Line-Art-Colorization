import torch.nn as nn
import torch
import math
from torch.nn.utils import spectral_norm

class DiscriminatorBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel_size=3,stride=1,padding=1,spec=True):
        super(DiscriminatorBlock, self).__init__()
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False))
            if spec 
            else nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):

    def __init__(self, spec=True):
        super(Discriminator, self).__init__()
        
        self.layer = nn.Sequential(
            DiscriminatorBlock(4, 16, 4, 2, 1, spec=spec), # 256 -> 128
            DiscriminatorBlock(16, 32, 4, 2, 1, spec=spec), # 128 -> 64
            DiscriminatorBlock(32, 64, 4, 2, 1, spec=spec), # 64 -> 32
            DiscriminatorBlock(64, 128, 4, 2, 1, spec=spec), # 32 -> 16
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        return self.layer(x)


class ResBlock(nn.Module):
    
    def __init__(self,  in_ch, out_ch, spec=True):
        super(ResBlock, self).__init__()
        self.residual = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
            if spec 
            else nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(out_ch, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
            if spec 
            else nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(out_ch, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return self.residual(x) + x

class ResBlockNet(nn.Module):
    
    def __init__(self, in_ch, out_ch, spec=False):
        super(ResBlockNet, self).__init__()
        self.layers = nn.Sequential(
            ResBlock(in_ch, out_ch, spec=spec),
            ResBlock(out_ch, out_ch, spec=spec),
            ResBlock(out_ch, out_ch, spec=spec),
            ResBlock(out_ch, out_ch, spec=spec),   
        )

    def forward(self, x):
        return self.layers(x) + x


class EncoderBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, spec=True):
        super(EncoderBlock, self).__init__()
        
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False))
            if spec 
            else nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):

    def __init__(self, in_ch=3, spec=False, LR=0.2):
        super(Encoder, self).__init__()
        
        self.layer1 = EncoderBlock(in_ch=in_ch, out_ch=16, spec=spec) # 256
        self.layer2 = EncoderBlock(in_ch=16, out_ch=16, spec=spec) # 256
        self.layer3 = EncoderBlock(in_ch=16, out_ch=32, spec=spec, stride=2) # 128
        self.layer4 = EncoderBlock(in_ch=32, out_ch=32, spec=spec) # 128
        self.layer5 = EncoderBlock(in_ch=32, out_ch=64, spec=spec, stride=2) # 64
        self.layer6 = EncoderBlock(in_ch=64, out_ch=64, spec=spec) # 64
        self.layer7 = EncoderBlock(in_ch=64, out_ch=128, spec=spec, stride=2) # 32
        self.layer8 = EncoderBlock(in_ch=128, out_ch=128, spec=spec) # 32
        self.layer9 = EncoderBlock(in_ch=128, out_ch=256, spec=spec, stride=2) # 16
        self.layer10 = EncoderBlock(in_ch=256, out_ch=256, spec=spec) # 16
        self.avg_down = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):
        # f: feature, df : down feature
        
        f1 = self.layer1(x)    # [2, 16, 256, 256]
        f2 = self.layer2(f1)   # [2, 16, 256, 256]
        f3 = self.layer3(f2)   # [2, 32, 128, 128]
        f4 = self.layer4(f3)   # [2, 32, 128, 128]
        f5 = self.layer5(f4)   # [2, 64, 64, 64]
        f6 = self.layer6(f5)   # [2, 64, 64, 64]
        f7 = self.layer7(f6)   # [2, 128, 32, 32]
        f8 = self.layer8(f7)   # [2, 128, 32, 32]
        f9 = self.layer9(f8)   # [2, 256, 16, 16]
        f10 = self.layer10(f9) # [2, 256, 16, 16]

        df1 = self.avg_down(f1)
        df2 = self.avg_down(f2)
        df3 = self.avg_down(f3)
        df4 = self.avg_down(f4)
        df5 = self.avg_down(f5)
        df6 = self.avg_down(f6)
        df7 = self.avg_down(f7)
        df8 = self.avg_down(f8)

        feature_list = [f9,f8,f7,f6,f5,f4,f3,f2,f1] #f10,
        
        out = torch.cat([df1,df2,df3,df4,df5,df6,df7,df8,f9,f10], dim=1)
        b, ch, h, w = out.size() # [2, 992, 16, 16]
        out = out.reshape((b, h * w, ch)) # [2, 256, 992]
        
        return out, feature_list


class DecoderBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,padding=1, down=True, use_drop=False, spec=True):
        super(DecoderBlock, self).__init__()
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch,kernel_size,stride,padding,bias=False))
            if down
            else spectral_norm(nn.ConvTranspose2d(in_ch, out_ch,kernel_size,stride,padding,bias=False)),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )
        self.use_drop=use_drop
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        out = self.layers(x)
        return self.dropout(out) if self.use_drop else out

class Decoder(nn.Module):
    
    def __init__(self, in_ch=992+992, features=256, spec=True):
        super(Decoder, self).__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_ch, features, 3, 1, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.dec5_1 = DecoderBlock(in_ch=992+992, out_ch=256, spec=spec)
        self.up4 = DecoderBlock(in_ch=512, out_ch=512, kernel_size=2, stride=2, padding=0, down=False, spec=spec)
        
        self.dec4_2 = DecoderBlock(in_ch=512+128, out_ch=128, spec=spec)
        self.dec4_1 = DecoderBlock(in_ch=128+128, out_ch=128, spec=spec)
        self.up3 = DecoderBlock(in_ch=128, out_ch=128, kernel_size=2, stride=2, padding=0, down=False, spec=spec)
        
        self.dec3_2 = DecoderBlock(in_ch=128+64, out_ch=64, spec=spec)
        self.dec3_1 = DecoderBlock(in_ch=64+64, out_ch=64, spec=spec)
        self.up2 = DecoderBlock(in_ch=64, out_ch=64, kernel_size=2, stride=2, padding=0, down=False, spec=spec)
        
        self.dec2_2 = DecoderBlock(in_ch=64+32, out_ch=32, spec=spec)
        self.dec2_1 = DecoderBlock(in_ch=32+32, out_ch=32, spec=spec)
        self.up1 = DecoderBlock(in_ch=32, out_ch=32, kernel_size=2, stride=2, padding=0, down=False, spec=spec)
        
        self.dec1_2 = DecoderBlock(in_ch=32+16, out_ch=16, spec=spec)
        self.dec1_1 = DecoderBlock(in_ch=16+16, out_ch=16, spec=spec)
        
        if spec:
            self.last_conv = spectral_norm(nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1))
        else:
            self.last_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x, feature_list):
        
        dec5_1 = self.initial_down(x) # [4, 256, 16, 16]
        up4 = self.up4(torch.cat((dec5_1, feature_list[0]),dim=1)) # [4, 512, 32, 32]
        
        dec4_2 = self.dec4_2(torch.cat((up4, feature_list[1]),dim=1)) # [4, 128, 32, 32]
        dec4_1 = self.dec4_1(torch.cat((dec4_2, feature_list[2]),dim=1)) # [4, 128, 32, 32]
        up3 = self.up3(dec4_1) # [4, 128, 64, 64]
        
        dec3_2 = self.dec3_2(torch.cat((up3, feature_list[3]),dim=1)) # [4, 64, 64, 64]
        dec3_1 = self.dec3_1(torch.cat((dec3_2, feature_list[4]),dim=1)) # [4, 64, 64, 64]
        up2 = self.up2(dec3_1) # [4, 64, 128, 128]
        
        dec2_2 = self.dec2_2(torch.cat((up2, feature_list[5]),dim=1)) # [4, 32, 128, 128]
        dec2_1 = self.dec2_1(torch.cat((dec2_2, feature_list[6]),dim=1)) # [4, 32, 128, 128]
        up1 = self.up1(dec2_1) # [4, 32, 256, 256]
        
        dec1_2 = self.dec1_2(torch.cat((up1, feature_list[7]),dim=1)) # [4, 16, 256, 256]
        dec1_1 = self.dec1_1(torch.cat((dec1_2, feature_list[8]),dim=1)) # [4, 16, 256, 256]
        
        out = self.last_conv(dec1_1) # [4, 3, 256, 256]
        
        return self.tanh(out)


class SCFT(nn.Module):

    def __init__(self, base=992):
        super(SCFT, self).__init__()
        self.w_q = nn.Linear(base, base)
        self.w_k = nn.Linear(base, base)
        self.w_v = nn.Linear(base, base)
        self.scailing_factor = math.sqrt(base)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, v_s, v_r):
        # v_s shape :  [2, 256, 992]
        # v_r shape :  [2, 256, 992]
        q = self.w_q(v_s) # [2, 256, 992]
        k = self.w_k(v_r) # [2, 256, 992]
        v = self.w_v(v_r) # [2, 256, 992]
        
        k_transpose = k.permute(0, 2, 1) # [2, 992, 256]
        attention_map = torch.bmm(q, k_transpose) # [2, 256, 256]
        attention_map = self.softmax(attention_map) / self.scailing_factor
        v_star = torch.bmm(attention_map, v) # [2, 256, 992]
        
        out = (v_s + v_star).permute(0, 2, 1) # [2, 992, 256]
        batch, ch, hw = out.size()
        out = out.reshape((batch, ch, 16, 16)) # [2, 992, 16, 16]
        return out, [q,k,v]

class Generator(nn.Module):

    def __init__(self, spec=True):
        super(Generator, self).__init__()
        self.Er = Encoder(in_ch=3, spec=spec)
        self.Es = Encoder(in_ch=1, spec=spec)
        self.decoder = Decoder(spec=spec)
        self.SCFT = SCFT(base=992)
        self.res_block = ResBlockNet(992, 992, False)

    def forward(self, reference, sketch):
        v_r, _ = self.Er(reference)
        v_s, feature_list = self.Es(sketch)
        v_c, q_k_v_list = self.SCFT(v_s, v_r)
        rv_c = self.res_block(v_c)
        out = self.decoder(torch.cat([rv_c, v_c], dim=1), feature_list)
        
        return out, q_k_v_list


if __name__ == '__main__':
    # test code
    Is = torch.randn(2,1,256,256)
    Ir = torch.randn(2,3,256,256)
    
    G = Generator()
    gen_out, _ = G(Ir, Is)
    print(f"gen_out : {gen_out.shape}")
    
    D = Discriminator()
    disc_out = D(torch.cat([gen_out, Is]))
    print(f"disc_out : {disc_out.shape}")