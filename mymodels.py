import torch
import torch.nn as nn
import os
import dataloader

__all__ = [
    'Color2Sketch', 'Sketch2Color', 'Discriminator', 
]

# Conv. block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, mode='Conv', norm=None, activation='relu', residual=False):
        super().__init__()
        # Convolution
        if mode=='Conv':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        elif mode=='Deconv':
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, 1, bias=bias)

        # Normalization
        self.norm = norm
        if norm=='BN':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm=='IN':
            self.norm = nn.InstanceNorm2d(out_channels)    
        elif norm=='GN':
            self.norm = nn.GroupNorm(32, out_channels)
        elif norm is None:
            self.norm = nn.Identity()

        # Activation
        if activation=='relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation=='lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation=='tanh':
            self.activation = nn.Tanh()
        elif activation=='sigmoid':
            self.activation = nn.Sigmoid()
        elif activation=='softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation is None:
            self.activation = nn.Identity()
            
        # Residual Connection
        self.residual = residual
        self.apply(weights_init)

    def forward(self, inputs):
        # Get input
        x = inputs
        
        # Residual connection
        if self.residual:
            residue = x

        # Convolution
        x = self.conv(x)
        
        # Normalization
        x = self.norm(x)
            
        # Activation
        x = self.activation(x)
        
        # Residual connection
        if self.residual:
            x = x + residue
        
        return x

class AttentionNet(nn.Module):
    def __init__(self, in_channels, out_channels, ngf, norm, activation):
        super(AttentionNet, self).__init__()
        self.layer1 = ConvBlock(in_channels=in_channels, 
                               out_channels=ngf,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               mode='Conv',
                               norm=norm,
                               activation=activation,)

        self.layer2 = ConvBlock(in_channels=ngf, 
                               out_channels=ngf*2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               mode='Conv',
                               norm=norm,
                               activation=activation,)           

        self.layer3 = ConvBlock(in_channels=ngf*2, 
                               out_channels=ngf,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               mode='Conv',
                               norm=norm,
                               activation=activation,)

        self.layer4 = ConvBlock(in_channels=ngf, 
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               mode='Conv',
                               norm=norm,
                               activation=activation,)               

    def forward(self, inputs): 
        x = inputs
        x = self.layer1(x)  
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
class AdaIn(nn.Module):
    def calc_mean_std(self, features):
        """
        :param features: shape of features -> [batch_size, c, h, w]
        :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
        """

        batch_size, c = features.size()[:2]
        features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
        features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
        return features_mean, features_std


    def forward(self, content_features, style_features):
        """
        Adaptive Instance Normalization
        :param content_features: shape -> [batch_size, c, h, w]
        :param style_features: shape -> [batch_size, c, h, w]
        :return: normalized_features shape -> [batch_size, c, h, w]
        """
        content_mean, content_std = self.calc_mean_std(content_features)
        style_mean, style_std = self.calc_mean_std(style_features)
        normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
        
        return normalized_features

class Encoder(nn.Module):
    def __init__(self, in_channels, ngf, norm, activation, return_mid=False):
        super(Encoder, self).__init__()
        # Build ResNet and change first conv layer to accept single-channel input
        self.layer1 = ConvBlock(in_channels=in_channels, 
                       out_channels=ngf,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       norm=None,
                       activation=activation,)

        self.layer2 = ConvBlock(in_channels=ngf, 
                       out_channels=ngf*2,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       norm=norm,
                       activation=activation,)

        self.layer3 = ConvBlock(in_channels=ngf*2, 
                       out_channels=ngf*4,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       norm=norm,
                       activation=activation,)

        self.return_mid = return_mid

    def forward(self, input_image):
        # Pass input through ResNet-gray to extract features
        x0 = input_image # nc * 256 * 256 
        x1 = self.layer1(x0) # 64 * 128 * 128 
        x2 = self.layer2(x1) # 128 * 64 * 64
        x3 = self.layer3(x2) # 256 * 32 * 32 

        if self.return_mid:
            return [x1, x2, x3]

        else:
            return x3

class Decoder(nn.Module):
    def __init__(self, out_channels=3, ngf=64, norm='BN', activation='relu'):
        super(Decoder, self).__init__()
        # Convolutional layers and upsampling     
        
        self.layer3 = ConvBlock(in_channels=ngf*4, 
                               out_channels=ngf*2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               mode='Deconv',
                               norm=norm,
                               activation=activation,)

        self.layer2 = ConvBlock(in_channels=ngf*2, 
                               out_channels=ngf,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               mode='Deconv',
                               norm=norm,
                               activation=activation,)             

        self.layer1 = ConvBlock(in_channels=ngf, 
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               mode='Deconv',
                               norm=norm,
                               activation='tanh',)            

    def forward(self, inputs): 
        x = inputs # 256 * 32 * 32
        x = self.layer3(x) # 128 * 64 * 64     
        x = self.layer2(x) # 64 * 128 * 128     
        x = self.layer1(x) # 3 * 256 * 256     

        return x

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, num_att, ngf=64, n_middle=9, norm='IN', activation='relu', pretrained=False):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.num_att = num_att
        self.encoder = Encoder(self.input_nc, ngf=ngf, norm=norm, activation=activation, return_mid=False)
        self.middle = []
        for i in range(n_middle):
            self.middle.append(ConvBlock(ngf*4, ngf*4, 3, 1, 1, norm=norm, activation=activation, residual=True))         
        self.middle = nn.Sequential(*self.middle)
        self.decoder = Decoder(output_nc*(num_att-1), ngf=ngf, norm=norm, activation=activation) 
        self.att = AttentionNet(3, num_att, ngf=ngf, norm=norm, activation=activation)

    def forward(self, inputs):
        img = inputs
        encode = self.encoder(img)
        feature = self.middle(encode)
        content = self.decoder(feature)
        att = self.att(img)
        
        att_list = []
        content_list = []
        
        temp_att = att[:, 0:1, :, :]
        output = inputs * temp_att
        content_list.append(inputs)
        att_list.append(temp_att)

        for i in range(1, self.num_att):
            temp_att = att[:, i:i+1, :, :]
            temp_content = content[:, self.output_nc*(i-1):self.output_nc*i, :, :]
            content_list.append(temp_content)
            att_list.append(temp_att)
            output = output + temp_content * temp_att

        return output, content_list, att_list
        
class Discriminator(nn.Module):
    def __init__(self, input_nc=3, norm='IN', activation='lrelu', pretrained=False):
        super(Discriminator, self).__init__()   
        self.layer1 = ConvBlock(in_channels=input_nc, 
                       out_channels=64,
                       kernel_size=4,
                       stride=2,
                       padding=1,
                       norm=norm,
                       activation=activation,)     
        
        self.layer2 = ConvBlock(in_channels=64, 
                       out_channels=128,
                       kernel_size=4,
                       stride=2,
                       padding=1,
                       norm=norm,
                       activation=activation,)             
        
        self.layer3 = ConvBlock(in_channels=128, 
                       out_channels=256,
                       kernel_size=4,
                       stride=2,
                       padding=1,
                       norm=norm,
                       activation=activation,)                     

        self.layer4 = ConvBlock(in_channels=256, 
                       out_channels=512,
                       kernel_size=4,
                       stride=1,
                       padding=1,
                       norm=norm,
                       activation=activation,)             
        
        self.layer5 = ConvBlock(in_channels=512, 
                       out_channels=1,
                       kernel_size=4,
                       stride=1,
                       padding=1,
                       norm=None,
                       activation=None)                    

    def forward(self, inputs):
        x = inputs
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.mean((2,3))

        return x
    
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
# To initialize model weights
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    else:
        pass
    
def toZeroThreshold(x, t=0.1):
    zeros = torch.cuda.FloatTensor(x.shape).fill_(0.0).cuda()
    return torch.where(x > t, x, zeros)