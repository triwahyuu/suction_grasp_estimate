## Converting PyTorch model to TorchScript model for C++ Inference
import sys
import os.path as osp
sys.path.append(osp.join('/'.join(osp.dirname(osp.abspath(__file__)).split('/')[:-1]), 'convnet'))

import argparse
import torch
from models.model import build_model

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
    parser.add_argument(
        '--output-path', default="", help='model path',
    )
    args = parser.parse_args()
    options = Options()
    rslt_path = osp.join(options.proj_path, "result") if args.output_path == "" else args.output_path
    
    print("loading model...")
    model = build_model(args.arch, options)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    torch.save({
        'epoch': checkpoint['epoch'],
        'iteration': checkpoint['iteration'],
        'arch': args.arch,
        'optim_state_dict': checkpoint['optim_state_dict'],
        'model_state_dict': checkpoint['model_state_dict'],
        'best_mean_iu': checkpoint['best_mean_iu'],
        'best_prec': checkpoint['best_prec'],
    }, args.checkpoint)

    # random input
    print("tracing model...")
    inp = torch.rand(1, 3, options.img_height, options.img_width)
    example = [inp, inp]

    # generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, (inp, inp))
    traced_script_module.save(osp.join(rslt_path, "model_converted.pt"))
    print("tracing done, model saved...")