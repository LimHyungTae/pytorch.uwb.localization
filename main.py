import os
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

import criteria
import utils
from dataloaders.params import INPUT_NAMES, OUTPUT_NAMES, INPUT_LEN
from models.multi_scale_ori import *
from dataloaders.Scaler import DataScaler
from dataloaders.datacontainer import ResultContainer
from metrics import *

args = utils.parse_command()
print(args)

# torch.cuda.empty_cache()

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'

fieldnames = ['rmse', 'mean', 'median', 'var', 'max']

scaledir = os.path.join(args.data, 'all')
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')

NUM_VAL_CSVS = len(os.listdir(valdir))
mm_scaler = DataScaler(scaledir)

def create_data_loaders(args, scaler):
    # Data loading code
    print("=> creating data loaders ...")

    train_loader = None
    val_loader = None

    if args.data in ["uwb_dataset"]:
        from dataloaders.uwb_dataloader import UWBDataloader
        # from dataloaders.dataloader import MyDataloader as UWBDataloader
        if not args.evaluate:
            train_dataset = UWBDataloader(traindir, 'train', scaler, args.y_target, seq_len=args.seq_len,
                                          stride=args.x_stride, interval=args.x_interval)
        val_dataset = UWBDataloader(valdir, 'val', scaler, args.y_target, seq_len=args.seq_len,
                                    stride=args.x_stride, interval=args.x_interval)
    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be included in data_names declared at parse_command() in utils.py')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader

def main():
    global args, output_directory, train_csv, test_csvs, mm_scaler
    # MinMax-Scaler!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # evaluation mode
    start_epoch = 0
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no best model found at '{}'".format(args.evaluate)
        print("=> loading best model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        output_directory = os.path.dirname(args.evaluate)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        _, val_loader = create_data_loaders(args, mm_scaler)
        args.evaluate = True
        validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
        return

    # optionally resume from a checkpoint
    elif args.resume:
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), \
            "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint '{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_data_loaders(args, mm_scaler)
        args.resume = True

    # create new model
    else:
        train_loader, val_loader = create_data_loaders(args, mm_scaler)
        print("=> creating Model ({}) ...".format(args.arch))
        from models.rnn_model import Model
        if args.arch == 'LSTM':
            model = Model(input_dim=args.x_dim, hidden_dim=args.hidden_size, Y_target=args.y_target, model_type="lstm")
        elif args.arch == 'GRU':
            model = Model(input_dim=args.x_dim, hidden_dim=args.hidden_size, Y_target=args.y_target, model_type="gru")
        if args.arch == 'RNN':
            model = Model(input_dim=args.x_dim, hidden_dim=args.hidden_size, Y_target=args.y_target, model_type="rnn")
        print("=> model created.")

        model_parameters = list(model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Num. of parameters: ", params)

        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

        # model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
        model = model.cuda()


    criterion = nn.MSELoss().cuda()
    # create results folder, if not already exists
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csvs = []
    for i in range(NUM_VAL_CSVS):
        test_csv_name = 'test_' + str(i) + '.csv'
        test_csv_each = os.path.join(output_directory, test_csv_name)
        test_csvs.append(test_csv_each)
    test_csv_total = os.path.join(output_directory, 'test.csv')
    test_csvs.append(test_csv_total)

    # 1 indicates total
    assert NUM_VAL_CSVS + 1 == len(test_csvs), "Something's wrong!"

    # create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[])
            writer.writeheader()
        for test_csv in test_csvs:
            with open(test_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    best_rmse = 1000000000

    print("=> Learning start.")
    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args.lr, args.decay_rate, args.decay_step)
        print("=> On training...")
        train(train_loader, model, criterion, optimizer, epoch)  # train for one epoch
        if epoch % args.validation_interval == 0:
            print("=> On validating...")
            result_rmse, results_list = validate(val_loader, model, epoch)  # evaluate on validation set
            # Save validation results
            print("=> On drawing results...")
            pngname = os.path.join(output_directory, str(epoch).zfill(2) + "_"
                                   + str(round(result_rmse, 5)) + ".png")
            utils.plot_trajectory(pngname, results_list[:-1])
            is_best = best_rmse > result_rmse
            if is_best:
                best_rmse = result_rmse
                best_name = os.path.join(output_directory, "best.csv")
                with open(best_name, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for result_container in results_list:
                        avg = result_container.result
                        writer.writerow({'rmse': avg.rmse, 'mean': avg.mean,
                                 'median': avg.median, 'var': avg.var, 'max': avg.error_max})

                    writer.writerow({'rmse': epoch, 'mean': 0,
                                     'median': 0, 'var': 0, 'max': 0})

                utils.save_output(results_list, epoch, output_directory)
                utils.save_checkpoint({
                    'args': args,
                    'epoch': epoch,
                    'arch': args.arch,
                    'model': model,
                    'optimizer': optimizer,
                    'scaler': mm_scaler
                }, is_best, epoch, output_directory)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()  # switch to train mode
    end = time.time()
    train_loss = 0

    average_meter = AverageMeter()

    for batch_idx, (x, y_gt, _) in enumerate(train_loader):

        x, y_gt = x.cuda(), y_gt.cuda()

        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        y_pred = model(x)

        loss = criterion(y_pred, y_gt)
        optimizer.zero_grad()
        loss.backward()  # compute gradient and do SGD step

        optimizer.step()
        torch.cuda.synchronize()

        gpu_time = time.time() - end
        # measure accuracy and record loss

        train_loss += loss.item()

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: %d  | %d / %d  |  lr: %.8f'
            %(epoch, batch_idx + 1, len(train_loader), optimizer.param_groups[0]['lr']))

    # avg = average_meter.average()
    # with open(train_csv, 'a') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writerow({'rmse': avg.rmse, 'rel': avg.absrel,
    #                      'd1': avg.delta1, 'd2': avg.delta2, 'd3': avg.delta3})

def validate(val_loader, model, epoch, write_to_file=True):

    model.eval()

    end = time.time()

    # 0 ~ N-1: 0 ~ N-1th csv
    # Nth: total
    results_list = []
    for _ in range(NUM_VAL_CSVS+1):
        result = ResultContainer(args.y_target)
        results_list.append(result)

    count = 0
    squares = 0
    is_initial = True

    for i, (x, y_gt, csv_id) in enumerate(val_loader):
        x, y_gt = x.cuda(), y_gt.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        end = time.time()
        with torch.no_grad():
            y_pred = model(x)

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        end = time.time()

        # Unscale output
        if args.y_target == "all":
            y_pred = y_pred[:, -1, :]
            y_gt = y_gt[:, -1, :]
        y_pred_unscaled = mm_scaler.undo_scale(y_pred.data.cpu())
        y_gt_unscaled = mm_scaler.undo_scale(y_gt.data.cpu())

        # Set result
        result = Result()
        result.evaluate(y_pred_unscaled, y_gt_unscaled)

        # Accumulate its trajectory
        results_list[csv_id].accum(y_pred_unscaled, y_gt_unscaled)

        results_list[csv_id].avg_meter.update(result, gpu_time, data_time, x.size(0))
        results_list[-1].avg_meter.update(result, gpu_time, data_time, x.size(0))

        if (i + 1) % args.print_freq == 0:
            avg = results_list[-1].avg_meter.average()
            print('%d / %d  |  RMSE: %.6f MEAN: %.6f MEDIAN: %.4f'
                  % (i, len(val_loader), avg.rmse, avg.mean, avg.median))

    rmse_final = None
    if write_to_file:
        for i_th_idx, test_csv in enumerate(test_csvs):
            metric = results_list[i_th_idx].result
            if i_th_idx < NUM_VAL_CSVS:
                gt_np, pred_np = results_list[i_th_idx].get_result()
            elif i_th_idx == NUM_VAL_CSVS: ## For total evaluation
                gt_np, pred_np = results_list[0].get_result()
                for k in range(1, NUM_VAL_CSVS):
                    gt_np_tmp, pred_np_tmp = results_list[k].get_result()
                    gt_np = np.concatenate((gt_np, gt_np_tmp), axis=0)
                    pred_np = np.concatenate((pred_np, pred_np_tmp), axis=0)
            else:
                raise RuntimeError("Not implemented!!!")

            metric.evaluate(pred_np, gt_np)
            with open(test_csv, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'rmse': metric.rmse, 'mean': metric.mean,
                                 'median': metric.median, 'var': metric.var, 'max': metric.error_max})
            if i_th_idx == NUM_VAL_CSVS:
                rmse_final = metric.rmse
                print("Final RMSE is ", rmse_final)

    return rmse_final, results_list

if __name__ == '__main__':
    main()
