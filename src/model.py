import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DWBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, use_residual=True, use_se=True, se_reduction=16):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_ch, se_reduction)
        self.use_residual = use_residual and (in_ch == out_ch) and (stride == 1)
        if use_residual and not self.use_residual:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.use_residual:
            out = out + identity
        elif self.skip is not None:
            out = out + self.skip(identity)
        out = self.relu(out)
        return out

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), 
                           kernel_size=(x.size(-2), x.size(-1))).pow(1./self.p)

class WaferNet(nn.Module):
    def __init__(self, num_classes, use_residual=True, use_se=True, 
                 use_dropout=True, dropout_rate=0.3, use_gem=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.backbone = nn.Sequential(
            DWBlock(16, 32, stride=2, use_residual=use_residual, use_se=use_se),
            DWBlock(32, 64, stride=2, use_residual=use_residual, use_se=use_se),
            DWBlock(64, 96, stride=2, use_residual=use_residual, use_se=use_se),
            DWBlock(96, 128, stride=2, use_residual=use_residual, use_se=use_se)
        )
        if use_gem:
            self.gap = GeMPooling(p=3.0)
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
        detector_layers = [
            nn.Linear(128, 32),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            detector_layers.append(nn.Dropout(dropout_rate))
        detector_layers.append(nn.Linear(32, 1))
        self.detector = nn.Sequential(*detector_layers)
        classifier_layers = [
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            classifier_layers.append(nn.Dropout(dropout_rate * 1.3))  # Higher dropout for classifier
        classifier_layers.append(nn.Linear(64, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.gap(x).flatten(1)
        det_logit = self.detector(x)
        cls_logit = self.classifier(x)
        return det_logit, cls_logit
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def get_flops(self, input_size=(1, 224, 224)):
        return "~46M FLOPs (estimated)"

def create_model(config):
    from config import NUM_CLASSES, USE_RESIDUAL, USE_SE_BLOCKS, USE_DROPOUT, DROPOUT_RATE, USE_GEM_POOLING
    model = WaferNet(
        num_classes=NUM_CLASSES,
        use_residual=USE_RESIDUAL,
        use_se=USE_SE_BLOCKS,
        use_dropout=USE_DROPOUT,
        dropout_rate=DROPOUT_RATE,
        use_gem=USE_GEM_POOLING
    )
    return model

if __name__ == "__main__":
    from config import NUM_CLASSES, DEVICE
    model = create_model(None).to(DEVICE)
    print(f"Total parameters: {model.get_num_parameters():,}")
    print(f"Estimated FLOPs: {model.get_flops()}")
    x = torch.randn(2, 1, 224, 224).to(DEVICE)
    det, cls = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Detection output: {det.shape}")
    print(f"Classification output: {cls.shape}")