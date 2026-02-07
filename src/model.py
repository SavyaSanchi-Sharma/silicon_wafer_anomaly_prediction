import torch.nn as nn
import torchvision.models as models

def get_model(num_classes, width_mult=0.75):
    model = models.mobilenet_v2(width_mult=width_mult)

    first_out_channels = int(32 * width_mult)

    model.features[0][0] = nn.Conv2d(
        1,
        first_out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    )

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
