## utilities code
from __future__ import division

import os
import math
import numpy as np

from skimage.transform import resize
from scipy.ndimage import gaussian_filter

from torchvision.transforms import ToTensor, Normalize, Resize
import torch


## prepare image input for inference
def prepare_input(color, depth, device):
    to_tensor = ToTensor()
    normalize = Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))

    color_img = normalize(to_tensor(color))
    color_img = color_img.view(1, color_img.size(0), color_img.size(1), color_img.size(2))

    depth_img = (np.asarray(depth, dtype=np.float64) / 8000).astype(np.float32).clip(0.0, 1.5)
    depth_img = np.stack([depth_img, depth_img, depth_img], axis=2)
    depth_img = normalize(to_tensor(depth_img))
    depth_img = depth_img.view(1, depth_img.size(0), depth_img.size(1), depth_img.size(2))

    return color_img.to(device), depth_img.to(device)


def post_process_output(out_tensor, options):
    cls_pred = out_tensor.data.max(1)[1].cpu().numpy().squeeze(0)
    pred = out_tensor.data.cpu().numpy().squeeze(0)[1]

    affordance = ((pred - pred.min()) / (pred.max() - pred.min()))
    affordance[~cls_pred.astype(bool)] = 0
    # affordance = gaussian_filter(affordance, 4)
    return cls_pred, affordance

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

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


def compute_precision(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    count = np.sum([1 for i in range(len(label)) if pred[i] == label[i]])
    return float(count) / len(label)

if __name__ == '__main__':
    pass