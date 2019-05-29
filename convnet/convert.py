## Converting PyTorch model to TorchScript model for C++ Inference

import argparse
import os.path as osp
import torch
from models.model import SuctionModel18, SuctionModel50
from models.model import SuctionRefineNet, SuctionRefineNetLW
from models.model import SuctionPSPNet

class Options:
    def __init__(self):
        p = osp.dirname(osp.abspath(__file__)).split('/')[:-1]
        self.proj_path = '/'.join(p)
        self.data_path = osp.join('/'.join(p[:-1]), 'dataset/')
        self.sample_path = osp.join(self.data_path, 'train-split.txt')
        self.img_height =  480
        self.img_width = 640
        self.n_class = 3
        self.output_scale = 8
        # available architecture: 
        # [resnet18, resnet34, resnet50, resnet101, rfnet50, rfnet101, pspnet18, pspnet101]
        self.arch = 'resnet18'


if __name__ == "__main__":
    model_choices = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'rfnet50', 'rfnet101', 'rfnet152', 'pspnet50', 'pspnet101', 'pspnet18', 'pspnet34']

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-a', '--arch', metavar='arch', default='resnet18', choices=model_choices,
        help='model architecture: ' + ' | '.join(model_choices) + ' (default: resnet18)'
    )
    parser.add_argument(
        '--checkpoint', required=True, help='model path',
    )
    args = parser.parse_args()
    options = Options()
    ckpt_path = p = osp.dirname(args.checkpoint)
    
    model = None
    backbone = 'resnet'
    if args.arch == 'resnet18' or args.arch == 'resnet34':
        model = SuctionModel18(options)
    elif args.arch == 'resnet50' or args.arch == 'resnet101' or args.arch == 'resnet152':
        model = SuctionModel50(options)
    elif args.arch == 'rfnet50' or args.arch == 'rfnet101' or args.arch == 'rfnet152':
        backbone = 'rfnet'
        model = SuctionRefineNetLW(options)
    elif args.arch == 'pspnet50' or args.arch == 'pspnet101' \
            or args.arch == 'pspnet18' or args.arch == 'pspnet34':
        model = SuctionPSPNet(options)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    # random input
    inp = torch.rand(1, 3, options.img_height, options.img_width)
    example = [inp, inp]

    # generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("model.pt")