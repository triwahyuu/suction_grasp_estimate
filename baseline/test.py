## Testing playground

import numpy as np
import torch
from torchvision import models

model = models.resnet18(pretrained=True)
## remove FC layers
new_model = torch.nn.Sequential(*(list(model.children())[:-3])) 

## to do: set batchnorm to fixed, solution:
# - set affine to false, then each `requires_grad` parameters to false;
# https://github.com/isht7/pytorch-deeplab-resnet/issues/39#issuecomment-419301286
# - set model's mode to eval (?)
# https://discuss.pytorch.org/t/freeze-batchnorm-layer-lead-to-nan/8385/2
# https://discuss.pytorch.org/t/proper-way-of-fixing-batchnorm-layers-during-training/13214/4

for m in new_model.children():
    if m.__class__.__name__ == 'BatchNorm2d':
        m.affine = False
    elif m.__class__.__name__ == 'Sequential':
        for n in m.children():
            if n.__class__.__name__ == 'BasicBlock':
                for k in n.children():
                    if k.__class__.__name__ == 'BatchNorm2d':
                        k.affine = False
                    elif k.__class__.__name__ == 'Sequential':
                        for l in k.children():
                            if l.__class__.__name__ == 'BatchNorm2d':
                                l.affine = False

print(new_model)