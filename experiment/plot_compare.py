import os
import argparse
import itertools

from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--result-path', default='', type=str, help='checkpoint path'
    )
    args = parser.parse_args()

    file_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(
        '/'.join(file_path.split('/')[:-1]), 'result_eval')
    result_path = args.result_path if args.result_path else result_path
    data = np.genfromtxt(os.path.join(result_path, 'summary.csv'),
                         delimiter=',', skip_header=1, dtype=str)
    resnet_fcn, resnet_psp, resnet_bise = [], [], []
    eff_fcn, eff_psp, eff_bise = [], [], []
    for row in data:
        if row[0].startswith('biseeffnet'):
            arch = row[0].replace('biseeffnet', 'efficientnet-')
            eff_bise.append([arch] + list(map(lambda x: np.float(x), row[1:])))
        elif row[0].startswith('bisenet'):
            arch = row[0].replace('bisenet', 'resnet')
            resnet_bise.append([arch] + list(map(lambda x: np.float(x), row[1:])))
        elif row[0].startswith('fcneffnet'):
            arch = row[0].replace('fcneffnet', 'efficientnet-')
            eff_fcn.append([arch] + list(map(lambda x: float(x), row[1:])))
        elif row[0].startswith('pspeffnet'):
            arch = row[0].replace('pspeffnet', 'efficientnet-')
            eff_psp.append([arch] + list(map(lambda x: float(x), row[1:])))
        elif row[0].startswith('pspnet'):
            arch = row[0].replace('pspnet', 'resnet')
            resnet_psp.append([arch] + list(map(lambda x: float(x), row[1:])))
        elif row[0].startswith('resnet'):
            resnet_fcn.append([row[0]] + list(map(lambda x: float(x), row[1:])))

    resnet = {'FCN': np.array(resnet_fcn, dtype=object),
              'PSPNet': np.array(resnet_psp, dtype=object),
              'BiSeNet': np.array(resnet_bise, dtype=object)}

    effnet = {'FCN': np.array(eff_fcn, dtype=object),
              'PSPNet': np.array(eff_psp, dtype=object),
              'BiSeNet': np.array(eff_bise, dtype=object)}

    colors = ['r', 'b', 'g']
    markers = ['o', 'v', 's']
    arches = ['FCN', 'PSPNet', 'BiSeNet']
    for backbone, bb_str in zip([resnet, effnet], ['ResNet', 'EfficientNet']):
        plt.figure()
        plt.xlim(0, 60)
        if(bb_str == 'ResNet'):
            plt.ylim(80.0, 90.5)
        elif(bb_str == 'EfficientNet'):
            plt.ylim(55.0, 90.5)
        for a, c, m in zip(arches, colors, markers):
            plt.plot(backbone[a][:, 5], backbone[a][:, 1]*100,
                     c + m + '-', markersize=7.5, label=a)

            chars = itertools.cycle('ABCDE')
            for row in backbone[a]:
                plt.text(row[5]-1.3, row[1]*100+0.22,
                         '['+next(chars)+']' , fontsize='medium', weight='bold')

        plt.legend(loc='lower right')
        plt.title('Segmentation Layer Perfomance on ' + bb_str + ' Backbone')
        plt.ylabel("precision (%)")
        plt.xlabel("inference time (ms)")

    plt.figure()
    plt.ylim(80.0, 90.0)
    bname = ['ResNet', 'EfficientNet']
    for backbone, c, m, b in zip([resnet, effnet], colors, markers, bname):
        plt.plot(backbone['PSPNet'][:, 5], backbone['PSPNet'][:, 1]*100,
                 c + m + '-', markersize=7.5, label=b)
        for row in backbone['PSPNet']:
            name = row[0].replace(b.lower(), '').replace('-', '')
            plt.text(row[5]-0.80, row[1]*100+0.18, name,
                     fontsize='medium', weight='bold')

    plt.legend(loc='lower right')
    plt.title('Backbone Comparison on PSPNet Segmentation Layer')
    plt.ylabel("precision (%)")
    plt.xlabel("inference time (ms)")

    plt.show()
