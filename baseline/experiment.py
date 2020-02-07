import os
from datetime import datetime
import shutil
import argparse

from evaluate import evaluate
import numpy as np


p = os.path.dirname(os.path.abspath(__file__)).split('/')
data_path = os.path.join('/'.join(p[:-2]), 'dataset/')
result_path = os.path.join('/'.join(p[:-2]), 'result/baseline/log/')
log_exp_prefix = 'log-'         # experiment log path prefix
res_exp_prefix = 'experiment-'  # experiment result file prefix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--datapath', dest='data_path', default='', help='suction grasp dataset path',
    )
    parser.add_argument(
        '-t','--no-frg-threshold', action='store_true', 
        help='disable experiment with the foreground threshold value'
    )
    parser.add_argument(
        '-r','--no-radius', action='store_true', 
        help='disable experiment with the radius of surface normals'
    )
    args = parser.parse_args()

    data_path = args.data_path if args.data_path != '' else data_path

    for exp_name in ['thres', 'rad']:
        if not os.path.exists(log_exp_prefix + exp_name):
            os.mkdir(log_exp_prefix + exp_name)
        f = open(res_exp_prefix + exp_name + '.txt', 'w+')
        f.write('timestamp,precision,recall,pred_time,color_threshold,depth_threshold,radius\n')
        f.close()

    if not args.no_radius:
        for rad in np.arange(0.04, 0.25, 0.02):
            now = datetime.now().strftime('%m%d_%H%M%S')
            result_log_name = os.path.join(log_exp_prefix + 'rad', 'result-' + now + '.txt')

            setting = [0.25, 0.1, round(rad, 2)]
            print("radius: %.2f" % (rad))
            
            result = evaluate(result_log_name, settings=setting)
            
            s = result.sum(axis=0)
            precision = s[0]/(s[0]+s[1])
            recall = s[0]/(s[0]+s[3])
            time_ave = s[4]/result.shape[0]
            with open(res_exp_prefix + 'rad.txt', 'a') as f:
                log = [now, precision, recall, time_ave] + setting
                log_str = "%s,%.8f,%.8f,%.8f,%.2f,%.2f,%.2f" % tuple(log)

                print(log_str)
                f.write(log_str + '\n')

    experiment_result = []
    if not args.no_frg_threshold:
        for col_thr in np.arange(0.15, 0.50, 0.05):
            for dep_thr in np.arange(0.05, 0.16, 0.01):
                now = datetime.now().strftime('%m%d_%H%M%S')
                result_log_name = os.path.join(log_exp_prefix + 'thres', 'result-' + now + '.txt')

                setting = list(map(lambda x: round(x, 2), [col_thr, dep_thr, 0.1]))
                print("color threshold: %.2f; depth threshold: %.2f" % (col_thr, dep_thr))
                
                result = evaluate(result_log_name, settings=setting)
                
                s = result.sum(axis=0)
                precision = s[0]/(s[0]+s[1])
                recall = s[0]/(s[0]+s[3])
                time_ave = s[4]/result.shape[0]
                with open(res_exp_prefix + 'thres.txt', 'a') as f:
                    log = [now, precision, recall, time_ave] + setting
                    log_str = "%s,%.8f,%.8f,%.8f,%.2f,%.2f,%.2f" % tuple(log)

                    # experiment_result.append(log)
                    print(log_str)
                    f.write(log_str + '\n')
    

    
    