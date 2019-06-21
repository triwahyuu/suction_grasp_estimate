__version__ = "0.1.0"
from .model import build_model

available_model = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'pspnet18', 'pspnet34', 'pspnet50', 'pspnet101',
    ## still failing to train:
    # 'bisenet18', 'bisenet34', 'bisenet50', 'bisenet101',
    # 'icnet18', 'icnet34', 'icnet50', 'icnet101',
    # 'rfnet50', 'rfnet101', 'rfnet152', 
]