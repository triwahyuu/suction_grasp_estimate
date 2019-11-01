from PIL import Image
import numpy as np
import tqdm

import os.path as osp
import argparse
import time
from datetime import datetime
import shutil

from predict import predict

import matplotlib.cbook
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

p = osp.dirname(osp.abspath(__file__)).split('/')
data_path = osp.join('/'.join(p[:-2]), 'dataset/')
result_path = osp.join('/'.join(p[:-2]), 'result/baseline/')
sample_list = open(data_path + 'test-split.txt').read().splitlines()

def evaluate(log_name, settings=[0.3, 0.02, 0.1], visualize=False, save_result=False):
    if visualize:
        plt.ion()
        plt.show()
        fig = plt.gcf()
        fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0, wspace=0.0, hspace=0.0)
        fig.canvas.set_window_title('Baseline Prediction Result')

    result = np.zeros((len(sample_list), 5))
    settings = list(map(lambda x: round(x, 2), settings))
    for n,fname in tqdm.tqdm(enumerate(sample_list), total=len(sample_list), desc='Processing', ncols=80):
        ## load the datasets
        rgb_in = Image.open(osp.join(data_path, "color-input/", fname + '.png'))
        rgb_bg = Image.open(osp.join(data_path, "color-background/", fname + '.png'))
        depth_in = Image.open(osp.join(data_path, "depth-input/", fname + '.png'))
        depth_bg = Image.open(osp.join(data_path, "depth-background/", fname + '.png'))
        cam_intrinsic = np.loadtxt(osp.join(data_path, "camera-intrinsics/", fname + '.txt'))

        ## get the suction affordance
        tm = time.perf_counter()
        surf_norm, score = predict(rgb_in, rgb_bg, depth_in, depth_bg, cam_intrinsic, settings)
        tm = time.perf_counter() - tm

        if save_result:
            score_im = Image.fromarray((score*255).astype(np.uint8))
            score_im.save(osp.join(result_path, fname + '.png'))

        ## Load ground truth manual annotations for suction affordances
        ## 0 - negative, 128 - positive, 255 - neutral (no loss)
        label = Image.open(data_path + 'label/' + fname + '.png')
        label_np = np.asarray(label, dtype=np.uint8)

        ## Suction affordance threshold
        ## take the top 1% prediction
        # threshold = score.max() - 0.0001
        threshold = np.percentile(score, 99)
        score_norm = (score*255).astype(np.uint8)
        sum_tp = np.sum(np.logical_and((score > threshold), (label_np == 128)).astype(np.int))
        sum_fp = np.sum(np.logical_and((score > threshold), (label_np == 0)).astype(np.int))
        sum_tn = np.sum(np.logical_and((score <= threshold), (label_np == 0)).astype(np.int))
        sum_fn = np.sum(np.logical_and((score <= threshold), (label_np == 128)).astype(np.int))
        
        result[n,:] = [sum_tp, sum_fp, sum_tn, sum_fn, tm]
        
        with open(osp.join('result/', log_name), 'a') as f:
            precision = 0 if (sum_tp + sum_fp) == 0 else sum_tp/(sum_tp + sum_fp)
            recall = 0 if (sum_tp + sum_fn) == 0 else sum_tp/(sum_tp + sum_fn)
            res_str = [fname, "%.8f"%(precision), "%.8f"%(recall), sum_tp, sum_fp, sum_tn, sum_fn, "%.8f"%(tm)]
            f.write(','.join(map(str, res_str)) + '\n')

        ## visualize
        if visualize:
            rgb_in_np = np.asarray(rgb_in, dtype=np.uint8)
            plt.subplot(1,3,1)
            plt.imshow(rgb_in_np)
            plt.yticks([]); plt.xticks([])
            plt.subplot(1,3,2)
            plt.imshow(score)
            plt.yticks([]); plt.xticks([])
            plt.subplot(1,3,3)
            plt.imshow(label_np)
            plt.yticks([]); plt.xticks([])
            plt.draw()
            plt.pause(0.01)
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--datapath', dest='data_path', default='', help='suction grasp dataset path',
    )
    parser.add_argument(
        '-v','--visualize', action='store_true', help='visualize result (pls don\'t use this)'
    )
    parser.add_argument(
        '-s','--save-result', action='store_true', help='save prediction result'
    )
    args = parser.parse_args()

    data_path = args.data_path if args.data_path != '' else data_path
    result_log_name = 'result-' + datetime.now().strftime('%m%d_%H%M') + '.txt'

    result = evaluate(result_log_name, visualize=args.visualize, save_result=args.save_result)
    
    ## save the result and calculate overal precision and recall
    shutil.copy(osp.join('result/', result_log_name), osp.join(result_path, 'log/'))
    s = result.sum(axis=0)
    precision = s[0]/(s[0]+s[1])
    recall = s[0]/(s[0]+s[3])
    time_ave = s[4]/result.shape[0]
    print(precision, recall, time_ave)