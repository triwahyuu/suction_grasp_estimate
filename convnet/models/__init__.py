__version__ = "0.1.0"
from .model import build_model

available_model = [] + \
    [('resnet' + str(n)) for n in [18, 34, 50, 101, 152]] + \
    [('pspnet' + str(n)) for n in [18, 34, 50, 101, 152]] + \
    [('fcneffnetb' + str(n)) for n in range(6)] + \
    [('pspeffnetb' + str(n)) for n in range(6)]
    ## still failing to train:
    # 'bisenet18', 'bisenet34', 'bisenet50', 'bisenet101',
    # 'icnet18', 'icnet34', 'icnet50', 'icnet101',
    # 'rfnet50', 'rfnet101', 'rfnet152', 