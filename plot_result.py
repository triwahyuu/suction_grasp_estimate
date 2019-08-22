import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--result-path', default='', type=str, help='checkpoint path'
    )
    parser.add_argument(
        '--arch', default='', type=str, help='checkpoint path'
    )
    args = parser.parse_args()


    file_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join('/'.join(file_path.split('/')[:-1]), 'result')
    result_path = args.result_path if args.result_path else result_path
    ave_const = 735 ## average each epoch

    list_dir = [a for a in sorted(os.listdir(result_path))[:-1] if os.path.isdir(os.path.join(result_path, a))]
    y = np.array([], dtype=np.float64)
    y_val = np.array([], dtype=np.float64)
    train_loss = np.array([], dtype=np.float64)
    val_loss = np.array([], dtype=np.float64)
    for d in list_dir:
        p = os.path.join(result_path, d, 'log.csv')
        data = np.genfromtxt(p, delimiter=',', skip_header=1, dtype=np.float)

        y = np.append(y, np.array([a for a in data[:,5].astype(np.float64) if not np.isnan(a)]))
        y_val = np.append(y_val, np.array([b for b in data[:,10] if not np.isnan(b)]))
        train_loss = np.append(train_loss, np.array([a for a in data[:,2].astype(np.float64) if not np.isnan(a)]))
        val_loss = np.append(val_loss, np.array([a for a in data[:,7].astype(np.float64) if not np.isnan(a)]))

    # get average
    y_ave = np.array([np.mean(y[i*ave_const:(i+1)*ave_const]) for i in range(y.shape[0]//ave_const)])
    loss_tr_ave = np.array([np.mean(train_loss[i*ave_const:(i+1)*ave_const]) for i in range(train_loss.shape[0]//ave_const)])
    x_ave = np.arange(y_ave.shape[0]).astype(np.int)
    x_val = np.arange(y_val.shape[0]).astype(np.int)

    loss_max = loss_tr_ave.max() if loss_tr_ave.max() > val_loss.max() else val_loss.max()
    loss_min = loss_tr_ave.min() if loss_tr_ave.min() < val_loss.min() else val_loss.min()
    loss_lim = (loss_min-loss_max*0.025, loss_max+loss_max*0.025)
    iou_max = y_ave.max() if y_ave.max() > y_val.max() else y_val.max()
    iou_min = y_ave.min() if y_ave.min() < y_val.min() else y_val.min()
    iou_lim = (iou_min-iou_max*0.025, iou_max+iou_max*0.025)

    fig1 = plt.figure(1)
    fig1.canvas.set_window_title("Mean IoU")
    plt.plot(x_ave, y_ave, label="Training")
    plt.plot(x_val, y_val, label="Validation")
    plt.ylim(iou_lim)
    plt.title("%s Mean IoU" % (args.arch))
    plt.legend()
    
    fig2 = plt.figure(2)
    fig2.canvas.set_window_title("Loss")
    plt.plot(x_ave, loss_tr_ave, label="Training")
    plt.plot(x_val, val_loss, label="Validation")
    plt.ylim(loss_lim)
    plt.title('%s Loss' % (args.arch))
    plt.legend()
    plt.show()
