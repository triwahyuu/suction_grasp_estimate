## utilities code
from __future__ import division

import os
import math
import numpy as np

from torchvision.transforms import ToTensor, Normalize, Resize
import torch


## prepare image input for inference
def prepare_input(color, depth, device):
    to_tensor = ToTensor()
    normalize = Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))

    color_img = normalize(to_tensor(color))
    color_img = color_img.view(1, color_img.size(0), color_img.size(1), color_img.size(2))

    depth = (to_tensor(np.asarray(depth, dtype=np.float32)) * 65536/10000).clamp(0.0, 1.2)
    depth_img = torch.cat([depth, depth, depth], 0)
    depth_img = normalize(depth_img)
    depth_img = depth_img.view(1, depth_img.size(0), depth_img.size(1), depth_img.size(2))

    img_input = [color_img.to(device), depth_img.to(device)]
    return img_input


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

## get label acuracy score
## https://github.com/foolwood/deepmask-pytorch/blob/master/utils/
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


if __name__ == '__main__':
    pass