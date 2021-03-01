import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from dataloaders.params import INPUT_NAMES, OUTPUT_NAMES

def parse_command():
    model_names = ['RNN', 'GRU', 'LSTM', "Bi-LSTM"]
    loss_names = ['l1', 'l2']
    data_names = ["uwb_dataset"]

    import argparse
    parser = argparse.ArgumentParser(description='UWB_LSTM')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='LSTM', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: LSTM)')
    parser.add_argument('--memo', default='0301', type=str)
    parser.add_argument('--data', metavar='DATA', default='uwb_dataset', choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: original one, which I acquired)')
    ########## RNN Params ##########
    parser.add_argument('-h_s', '--hidden-size', default=256, type=int, help='Hidden size')
    parser.add_argument('--seq-len', '-sl', default=12, type=int, metavar='SEQLENGTH',
                        help='Sequence length for input')
    parser.add_argument('--x-stride', default=1, type=int, metavar='ST',
                        help='Stride between each UWB sample in UWB samples(default: 1)')
    parser.add_argument('--x-dim', default=8, type=int, metavar='DIM',
                        help='Input dimension (default: 8 since the number of UWB is 8)')
    parser.add_argument('--x-interval', default=1, type=int, metavar='I',
                        help='Interval of each datum in UWB samples (default: 1)')
    parser.add_argument('--y-target', default="all", type=str, metavar='OUTPUT_target', choices=["end", "all"],
                        help='When calculating loss function!')
    ########## Learning Params ##########
    parser.add_argument('--gpu', default="1")
    parser.add_argument('-v', '--validation-interval', default=1, type=int, metavar='VI',
                        help='Validation interval. The number of data loading workers (default: 10)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')
    parser.add_argument('-b', '--batch-size', default=3000, type=int, help='mini-batch size')
    parser.add_argument('--decay-rate', default=0.7, type=float, metavar='dr',
                        help='number of decay_step (default: 0.2)')
    parser.add_argument('--decay-step', default=5, type=int, metavar='ds',
                        help='number of decay_step (default: 5)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate (default 0.001)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=300, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str, default='',
                        help='evaluate model on validation set')
    parser.add_argument('--no-pretrain', dest='pretrained', action='store_false',
                        help='not to use ImageNet pre-trained weights')

    parser.set_defaults(pretrained=True)
    args = parser.parse_args()

    return args

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


def save_output(results_list, epoch, output_directory):
    output_filename = os.path.join(output_directory, 'output-' + str(epoch) + '.pth.tar')
    torch.save({"output": results_list}, output_filename)


def adjust_learning_rate(optimizer, epoch, lr_init, decay_rate, decay_step):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (decay_rate ** (epoch // decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_output_directory(args):
    output_directory = os.path.join('results',
        '{}_{}.y_target={}.seqlen={}.interal={}.stride={}.arch={}..criterion={}.lr={}.dr={}.ds={}.bs={}'.
        format(args.memo, args.data, args.y_target, args.seq_len, args.x_interval, args.x_stride,
            args.arch, args.criterion, args.lr, args.decay_rate, args.decay_step, args.batch_size))

    if output_directory.split("/")[1] in os.listdir('results'):
        from random import random
        output_directory = output_directory + "_" + str(int(random() * 10000))
    return output_directory


def calc_RMSE(scaler, y_gt, y_pred):
    y_gt_cpu = y_gt.cpu().detach().numpy()
    y_pred_cpu = y_pred.cpu().detach().numpy()

    y_gt_unscaled = scaler.undo_scale(y_gt_cpu)
    y_pred_unscaled = scaler.undo_scale(y_pred_cpu)

    return np.sqrt(np.mean((y_gt_unscaled - y_pred_unscaled) ** 2))

def plot_trajectory(png_name, result_containers):
    # For alpha
    MIN_VALUES = [-2.7, -2.7, -2.7, -2.7]
    MAX_VALUES = [2.7, 2.7, 2.7, 2.7]


    len_col = 4
    len_row = ceil(len(result_containers) / len_col)
    plt.figure(figsize=(22, 5))
    for i, result_container in enumerate(result_containers):

        plt.subplot(len_row, len_col, i + 1)
        y_gt_set, y_pred_set = result_container.trajectory_container.get_results()
        len_x = y_gt_set.shape[0]
        plt.plot(y_gt_set[:, 0], y_gt_set[:, 1], 'g', linestyle='-', label='gt')
        plt.plot(y_pred_set[:, 0], y_pred_set[:, 1], 'b', linestyle='--', label='pred')

        plt.grid()
        plt.xlim(MIN_VALUES[i], MAX_VALUES[i])
        plt.ylim(MIN_VALUES[i], MAX_VALUES[i])
        plt.legend()
        plt.rcParams["legend.loc"] = 'lower left'

    fig = plt.gcf()
    fig.savefig(png_name)
    plt.close('all')


if __name__ == "__main__":
    abs_dir = "/home/shapelim/ws/kari-lstm/uwb_dataset/val"
    csv_names = sorted(os.listdir("/home/shapelim/ws/kari-lstm/uwb_dataset/val"))
    plt.figure(figsize=(20, 5))
    for i, csvname in enumerate(csv_names):
        fname = os.path.join(abs_dir, csvname)
        y_gt_set = np.loadtxt(fname, delimiter=',')[:, -2:]
        plt.subplot(1, 4, i + 1)
        plt.plot(y_gt_set[:, 0], y_gt_set[:, 1], 'g', linestyle='-', label='gt')
        plt.grid()
        plt.legend()

    fig = plt.gcf()
    fig.savefig("debug.png")
