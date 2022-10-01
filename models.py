# =============================================================================
# Install necessary packages
# =============================================================================
# pip install inplace-abn
# pip install timm


# =============================================================================
# Import required libraries
# =============================================================================
from torch import nn
import torchvision
import timm


class ResNet101(nn.Module):
    def __init__(self, args, num_classes, pretrained):
        super(ResNet101, self).__init__()
        self.path = args.save_dir + 'ResNet101_Corel-5k.pth'

        resnet = torchvision.models.resnet101(pretrained=pretrained)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=num_classes)
        )
        # resnet.avgpool = nn.AdaptiveMaxPool2d(1)
        self.net = resnet

    def forward(self, img):
        return self.net(img)


class ResNeXt50(nn.Module):
    def __init__(self, args, num_classes, pretrained):
        super(ResNeXt50, self).__init__()
        self.path = args.save_dir + 'ResNext50_Corel-5k.pth'
        
        resnext = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        resnext.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=num_classes)
        )
        # resnext.avgpool = nn.AdaptiveMaxPool2d(1)
        self.net = resnext

    def forward(self, img):
        return self.net(img)


class Xception(nn.Module):
    def __init__(self, args, num_classes, pretrained):
        super(Xception, self).__init__()
        self.path = args.save_dir + 'Xception_Corel-5k.pth'

        xception = timm.create_model(
            'xception', pretrained=pretrained, global_pool='avg')
        xception.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=num_classes)
        )
        self.net = xception

    def forward(self, img):
        return self.net(img)


class TResNet(nn.Module):
    def __init__(self, args, num_classes, pretrained):
        super(TResNet, self).__init__()
        self.path = args.save_dir + 'TResNet_Corel-5k.pth'

        tresnet = timm.create_model('tresnet_m', pretrained=pretrained)
        tresnet.head.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=num_classes)
        )
        self.net = tresnet

    def forward(self, x):
        return self.net(x)
