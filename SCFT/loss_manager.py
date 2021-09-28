import torch
import torchvision
from torch import nn


class VGGPerceptualLoss(torch.nn.Module):
    """
    https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    """
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        
        blocks.append(torchvision.models.vgg19(pretrained=True).features[:5].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[5:10].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[10:19].eval())
        blocks.append(torchvision.models.vgg19(pretrained=True).features[19:27].eval())
        # blocks.append(torchvision.models.vgg19(pretrained=True).features[28:36].eval())
        
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        
        input = (input-self.mean) / self.std
        input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        
        target = (target-self.mean) / self.std
        target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        perceptual_loss = 0.0
        style_loss = 0.0
        
        x, y = input, target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            perceptual_loss += torch.nn.functional.l1_loss(x, y)
            style_loss += torch.nn.functional.l1_loss(self.gram_matrix(x), self.gram_matrix(y))
        return perceptual_loss, style_loss

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


class SCTFLoss:
    
    def __init__(self, args, adversarial_loss):
        self.args = args
        self.vgg_loss = VGGPerceptualLoss().to(args.cuda).eval()
        self.adversarial_loss = adversarial_loss
    
    # Discrimintor Loss
    def disc_adversarial_loss(self, fake_y, real_y, label_smoothing=0.9):
        
        valid = torch.ones_like(real_y)
        fake = torch.zeros_like(fake_y)
        
        disc_real = self.adversarial_loss(real_y, label_smoothing*valid)
        disc_fake = self.adversarial_loss(fake_y, fake)
        
        return disc_real, disc_fake
    
    # Generative Loss
    def gen_adversarial_loss(self, fake_y, label_smoothing=0.9):
        
        valid = torch.ones_like(fake_y)
        gen_loss = self.adversarial_loss(fake_y, label_smoothing*valid)
        return gen_loss
        
    
    def reconstruct_loss(self, fake_y, real_y):
        l1 = nn.L1Loss()
        reconstruct_loss = l1(fake_y, real_y)
        
        return reconstruct_loss
        
    def perceptron_and_style_loss(self, fake_y, real_y):
        preceptron_loss, style_loss = self.vgg_loss(fake_y, real_y)
        
        return preceptron_loss, style_loss
    
    
    def triplet_loss(self, qkv_list):
        
        def _triplet_loss(anchor, positive, negative, margin):
            triplet = nn.TripletMarginLoss(margin=margin)    
            triplet_loss = triplet(anchor, positive, negative)
            
            return triplet_loss
        
        anchor = qkv_list[0].contiguous().view(self.args.batch_size, -1)
        positive = qkv_list[1].contiguous().view(self.args.batch_size, -1)
        negative = qkv_list[2].contiguous().view(self.args.batch_size, -1)
        
        return _triplet_loss(anchor, positive, negative, self.args.weight_g_triplet)