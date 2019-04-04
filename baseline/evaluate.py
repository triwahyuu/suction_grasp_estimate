from PIL import Image
import numpy as np

if __name__ == "__main__":
    data_path = '/home/tri/skripsi/dataset/'
    code_path = '/home/tri/skripsi/suction_grasp_estimate/'
    
    n = 0
    df = open(data_path + 'test-split.txt')
    data = df.read().splitlines()
    result = np.zeros((len(data), 4))
    for fname in data:
        print(fname, end='\t')

        ## get the baseline result and ground truth label
        result = Image.open(data_path + 'baseline/' + fname + '.png')
        label = Image.open(data_path + 'label/' + fname + '.png')

        ## calculate the precision and recall
        result_np = np.asarray(result, dtype=np.uint8)
        label_np = np.asarray(label, dtype=np.uint8)
        threshold = np.percentile(result_np, 99)
        sum_tp = np.sum(np.logical_and((result_np > threshold), (label_np == 128)).astype(np.int))
        sum_fp = np.sum(np.logical_and((result_np > threshold), (label_np == 0)).astype(np.int))
        sum_tn = np.sum(np.logical_and((result_np <= threshold), (label_np == 0)).astype(np.int))
        sum_fn = np.sum(np.logical_and((result_np <= threshold), (label_np == 128)).astype(np.int))
        precision = sum_tp/(sum_tp + sum_fp)
        recall = sum_tp/(sum_tp + sum_fn)
        result[n,:] = [sum_tp, sum_fp, sum_tn, sum_fn]
        print("%.8f\t%.8f" % (precision, recall))
        n += 1
    
    s = result.sum(axis=0)
    precision = s[0]/(s[0]+s[1])
    recall = s[0]/(s[0]+s[3])
    print(precision, recall)
