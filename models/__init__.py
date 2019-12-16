__version__ = "0.1.0"
import models.model

available_model = [] + \
    [('resnet' + str(n)) for n in [18, 34, 50, 101, 152]] + \
    [('pspnet' + str(n)) for n in [18, 34, 50, 101, 152]] + \
    [('bisenet' + str(n)) for n in [18, 34, 50, 101]] + \
    [('fcneffnetb' + str(n)) for n in range(6)] + \
    [('pspeffnetb' + str(n)) for n in range(6)] + \
    [('biseeffnetb' + str(n)) for n in range(6)]
    ## still failing to train:
    # 'icnet18', 'icnet34', 'icnet50', 'icnet101',
    # 'rfnet50', 'rfnet101', 'rfnet152', 

def build_model(arch, n_class=3, out_size=(480, 640)):
    arch_nodigit = ''.join([a for a in arch if not a.isdigit()])
    arch2class_map = {
        'resnet': 'SuctionModelFCN',
        'pspnet': 'SuctionPSPNet',
        'bisenet' : 'SuctionBiSeNet',
        'fcneffnetb' : 'SuctionEffNetFCN',
        'pspeffnetb' : 'SuctionEffNetPSP',
        'biseeffnetb' : 'SuctionEfficientBiSeNet',
        # 'icnet' : 'SuctionICNet',
        # 'rfnet' : 'SuctionRefineNetLW'
    }
    try:
        module = getattr(models.model, arch2class_map[arch_nodigit])
    except KeyError:
        raise Exception('model %s is not supported' % (arch))
        
    m = module(arch=arch, n_class=n_class, out_size=out_size)
    return m