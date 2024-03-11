import torch 
from torch import nn
import torch.nn.functional as F

def my_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.2)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class AdaIn(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.lin = nn.Linear(style_dim, channels * 2)
        self.channels = channels
        
    def forward(self, x, s):
        proj_s = self.lin(s).unsqueeze(2).unsqueeze(3) # batch, style_dim -> batch, channels * 2, 1, 1
        bias = proj_s[:, :self.channels, :, :]
        mult = proj_s[:, self.channels:, :, :]
        return self.norm(x) * (1 + mult) + bias

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, upsample=False, use_adain=False, style_dim=None):
        super().__init__()
        self.use_adain = use_adain
        
        if downsample:
            self.resample = nn.AvgPool2d(2)
        elif upsample:
            self.resample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            self.resample = nn.Identity()
            
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        else:
            self.skip_conv = nn.Identity()
            
        if use_adain:
            self.norm1 = AdaIn(in_channels, style_dim)
            self.norm2 = AdaIn(in_channels, style_dim)
        else:
            self.norm1 = nn.InstanceNorm2d(in_channels)
            self.norm2 = nn.InstanceNorm2d(in_channels)
        
        self.activation = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        
    
    def forward(self, x, s=None):
        skip_res = self.resample(x)
        skip_res = self.skip_conv(skip_res)
        
        x = self.norm1(x, s) if self.use_adain else self.norm1(x)
        x = self.activation(x)
        x = self.resample(x)
        x = self.conv1(x)
        x = self.norm2(x, s) if self.use_adain else self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        
        res = (x + skip_res) / 1.41421
        return res


class StyleEncoder(nn.Module):
    def __init__(self, num_domains, style_dim, **kwargs):
        super().__init__()
        
        self.stack = nn.Sequential(
            nn.Conv2d(3, 64, 1),                # 256
            ResBlock(64, 128, downsample=True), # 128
            ResBlock(128, 256, downsample=True), # 64
            ResBlock(256, 512, downsample=True),  # 32
            ResBlock(512, 512, downsample=True), # 16
            ResBlock(512, 512, downsample=True), # 8
            ResBlock(512, 512, downsample=True), # 4
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4), # 1
            nn.LeakyReLU(0.2),
        )
        
        self.domain_heads = nn.ModuleList()
        for i in range(num_domains):
            self.domain_heads.append(nn.Linear(512, style_dim))
        
    def forward(self, x, y):
        common = self.stack(x).reshape(x.shape[0], 512)
        
        out = []
        for domain_head in self.domain_heads:
            out.append(domain_head(common))
        out = torch.stack(out, 1) # batch, num_domain, style_dim
        idx = torch.arange(y.shape[0], device=y.device)
        res = out[idx, y]  # batch, style_dim  
        return res
        

class Discriminator(nn.Module):
    def __init__(self, num_domains, **kwargs):
        super().__init__()
        
        self.stack = nn.Sequential(
            nn.Conv2d(3, 64, 1),                # 256
            ResBlock(64, 128, downsample=True), # 128
            ResBlock(128, 256, downsample=True), # 64
            ResBlock(256, 512, downsample=True),  # 32
            ResBlock(512, 512, downsample=True), # 16
            ResBlock(512, 512, downsample=True), # 8
            ResBlock(512, 512, downsample=True), # 4
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4), # 1
            nn.LeakyReLU(0.2),
        )
        
        self.domain_head = nn.Linear(512, num_domains)
        
    def forward(self, x, y):
        common = self.stack(x).reshape(x.shape[0], 512)
        res = self.domain_head(common)
        idx = torch.arange(y.shape[0], device=y.device)
        res = res[idx, y] 
        return res
    
    
class MappingNetwork(nn.Module):
    def __init__(self, num_domains, latent_dim, style_dim, **kwargs):
        super().__init__()
        
        self.stack = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
        )
        
        self.domain_heads = nn.ModuleList()
        for i in range(num_domains):
            self.domain_heads.append(nn.Sequential(
                nn.Linear(512, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, style_dim))
            )
    
    def forward(self, z, y):
        common = self.stack(z).reshape(z.shape[0], -1)
        
        out = []
        for domain_head in self.domain_heads:
            out.append(domain_head(common))
        out = torch.stack(out, 1) # batch, num_domain, style_dim
        idx = torch.arange(y.shape[0], device=y.device)
        res = out[idx, y]  # batch, style_dim  
        return res
    
    
class Generator(nn.Module):
    def __init__(self, style_dim, **kwargs):
        super().__init__()
        
        self.first_conv = nn.Conv2d(3, 64, 1)
        self.last_conv = nn.Conv2d(64, 3, 1)
        
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(ResBlock(64, 128, downsample=True)) # 256 -> 128
        self.down_blocks.append(ResBlock(128, 256, downsample=True)) # 128 -> 64
        self.down_blocks.append(ResBlock(256, 512, downsample=True)) # 64 -> 32
        self.down_blocks.append(ResBlock(512, 512, downsample=True)) # 32 -> 16
        self.down_blocks.append(ResBlock(512, 512, downsample=True)) # 16 -> 8
        self.down_blocks.append(ResBlock(512, 512)) # bottleneck 8
        self.down_blocks.append(ResBlock(512, 512)) # bottleneck 8
        
        self.up_blocks = nn.ModuleList()
        self.up_blocks.append(ResBlock(512, 512, use_adain=True, style_dim=style_dim)) # bottleneck 8
        self.up_blocks.append(ResBlock(512, 512, use_adain=True, style_dim=style_dim)) # bottleneck 8
        self.up_blocks.append(ResBlock(512, 512, upsample=True, use_adain=True, style_dim=style_dim)) # 8 -> 16
        self.up_blocks.append(ResBlock(512, 512, upsample=True, use_adain=True, style_dim=style_dim)) # 16 -> 32
        self.up_blocks.append(ResBlock(512, 256, upsample=True, use_adain=True, style_dim=style_dim)) # 32 -> 64
        self.up_blocks.append(ResBlock(256, 128, upsample=True, use_adain=True, style_dim=style_dim)) # 64 -> 128
        self.up_blocks.append(ResBlock(128, 64, upsample=True, use_adain=True, style_dim=style_dim)) # 128 -> 256
        
    def forward(self, x, s):
        x = self.first_conv(x)
        
        # chache = {32: None, 64: None, 128: None}
        chache = {32: None, 16: None}
        
        for i in range(len(self.down_blocks)):
            if x.shape[-1] in chache:
                chache[x.shape[-1]] = x
            x = self.down_blocks[i](x)
        
        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x, s)
            if x.shape[-1] in chache:
                x = (chache[x.shape[-1]] + x) / 1.41421
        
        x = self.last_conv(x)
        return x
            
        
        