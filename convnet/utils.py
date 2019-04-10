## utilities code
## https://github.com/foolwood/deepmask-pytorch/blob/master/utils/
from __future__ import division

import os
import logging
import math
import numpy as np
import matplotlib.pyplot as plt

logs = set()

class Filter:
    def __init__(self, flag):
        self.flag = flag
    def filter(self, x): return self.flag

def get_format(logger, level):
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])

        if level == logging.INFO:
            logger.addFilter(Filter(rank == 0))
    else:
        rank = 0
    format_str = '[%(asctime)s-rk{}-%(filename)s#%(lineno)3d] %(message)s'.format(rank)
    formatter = logging.Formatter(format_str)
    return formatter


def init_log(name, level = logging.INFO):
    if (name, level) in logs: return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = get_format(logger, level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def add_file_handler(name, log_file, level = logging.INFO):
    logger = logging.getLogger(name)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(get_format(logger, level))
    logger.addHandler(fh)


init_log('global')
# init_log('global', logging.WARN)


def print_speed(i, i_time, n):
    """print_speed(index, index_time, total_iteration)"""
    logger = logging.getLogger('global')
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    logger.info('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' % (i, n, i/n*100, average_time, remaining_day, remaining_hour, remaining_min))


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

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


def main():
    for i, lvl in enumerate([logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]):
        log_name = str(lvl)
        init_log(log_name, lvl)
        logger = logging.getLogger(log_name)
        print('****cur lvl:{}'.format(lvl))
        logger.debug('debug')
        logger.info('info')
        logger.warning('warning')
        logger.error('error')
        logger.critical('critiacal')


if __name__ == '__main__':
    main()