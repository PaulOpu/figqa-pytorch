import os
import os.path as pth
from itertools import islice
from timeit import default_timer as timer
from time import gmtime, strftime
from collections import defaultdict
import gc

import numpy as np
import tqdm
import time

import torch
import torch.optim
import torch.optim.lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader

import figqa.options
import figqa.utils as utils
import figqa.utils.visualize
from figqa.utils.datasets import FigQADataset, batch_iter


def chargrid_log(iter_idx,viz,model):

    #Decision Factor
    #dec_factor = model.module.de
    #viz.append_data(x, y, key, line_name)

    #CharGrid Net
    chargrid_net = model.module.chargrid_net

    name = "chargrid"
    for idx in [0,3,6]:
        weights = chargrid_net[idx].weight.data.cpu().numpy()
        gradients = chargrid_net[idx].weight.grad.cpu().numpy()
        viz.append_histogram(iter_idx, weights.reshape(-1), f"{name}_weights_{idx}")
        viz.append_histogram(iter_idx, gradients.reshape(-1), f"{name}_gradients_{idx}")

    #EntityGrid Net
    name = "entitygrid"
    entitygrid_net = model.module.entitygrid_net
    for idx in [0,3,6,9]:
        weights = entitygrid_net[idx].weight.data.cpu().numpy()
        gradients = entitygrid_net[idx].weight.grad.cpu().numpy()
        viz.append_histogram(iter_idx, weights.reshape(-1), f"{name}_weights_{idx}")
        viz.append_histogram(iter_idx, gradients.reshape(-1), f"{name}_gradients_{idx}")



def log_stuff(iter_idx, loss, batch, pred, val_dataloader, model,
              criterion, epoch, optimizer, running_accs, viz, args,
              **kwargs):
    global running_loss, start_t
    #######################################################################
    # report numbers on this train batch
    if iter_idx % 100 != 0:
        return
    # loss
    alpha = .70
    if running_loss is None:
        running_loss = loss.data.item()#[0]
    else:
        running_loss = alpha * running_loss + (1 - alpha) * loss.data.item()
    viz.append_data(iter_idx, running_loss, 'Loss', 'running loss')

    #Chargrid: Visualize Weight 
    chargrid_log(iter_idx,viz,model)

    # accuracy
    _, pred_idx = torch.max(pred, dim=1)
    correct = (batch['answer'] == pred_idx)
    train_acc = correct.cpu().data.numpy().mean()
    viz.append_data(iter_idx, train_acc, 'Acc', 'acc')

    # learning rate
    viz.append_data(iter_idx, optimizer.param_groups[0]['lr'], 'Learning rate', 'lr', ytype='log')

    # accuracy by question type
    for qtype, meta in enumerate(utils.QTYPE_ID_TO_META):
        qtype_mask = (batch['qtype'] == qtype)
        if qtype_mask.sum().data.item() != 0:
            qtype_correct = correct[qtype_mask]
            qtype_acc = qtype_correct.sum().data.item() / qtype_correct.size(0)
            running_accs[qtype] = 0.20 * qtype_acc + \
                                  (1 - 0.20) * running_accs[qtype]
        viz.append_data(iter_idx, running_accs[qtype],
                        'Train Question Type Acc', meta[0] + ' ' + str(meta[1]))

    # print to command line
    end_t = timer()
    time_stamp = strftime('%a %d %b %y %X', gmtime())
    t_diff = end_t - start_t
    log_line = ('[{time_stamp}][Ep: {epoch:0>2d}][Iter: {iter_idx}]'
                '[Time: {t_diff:.2f}][Loss: {running_loss:.4f}]')
    print(log_line.format(running_loss=running_loss, **locals()),flush=True)
    start_t = end_t

    #######################################################################
    # numbers on a few batches of val
    if iter_idx % 500 != 0:
        return
    val_batches = 10
    val_losses = []
    val_accs = []
    val_correct_by_qtype = {qtype: [] for qtype, _ in
                                            enumerate(utils.QTYPE_ID_TO_META)}

    for _, val_batch in islice(batch_iter(val_dataloader, args, volatile=True), val_batches):
        val_pred = model(val_batch)
        val_loss = criterion(val_pred, val_batch['answer']).cpu().data.numpy()
        val_losses.append(val_loss)
        _, val_pred_idx = torch.max(val_pred, dim=1)
        val_correct = (val_batch['answer'] == val_pred_idx)
        val_acc = val_correct.cpu().data.numpy().mean()
        val_accs.append(val_acc)
        # accuracy by question type
        for qtype, meta in enumerate(utils.QTYPE_ID_TO_META):
            qtype_mask = (val_batch['qtype'] == qtype)
            if qtype_mask.sum().data.item() == 0:
                continue
            qtype_correct = val_correct[qtype_mask]
            val_correct_by_qtype[qtype].append(qtype_correct)

    # plot stuff
    viz.append_data(iter_idx, np.mean(val_losses), 'Loss', 'val loss')
    viz.append_data(iter_idx, np.mean(val_accs), 'Acc', 'val acc')
    acc_per_chart_type = defaultdict(lambda: [])
    for qtype, meta in enumerate(utils.QTYPE_ID_TO_META):
        correct = sum(c.sum().data.item() for c in val_correct_by_qtype[qtype])
        total = sum(c.size(0) for c in val_correct_by_qtype[qtype])
        qtype_acc = correct / total if total > 0 else 0.5
        viz.append_data(iter_idx, qtype_acc, 'Val Question Type Acc',
                        meta[0] + ' ' + str(meta[1]))
        chart_type = meta[1]
        acc_per_chart_type[chart_type].append(qtype_acc)
    for chart_type in acc_per_chart_type:
        acc = np.mean(acc_per_chart_type[chart_type])
        viz.append_data(iter_idx, acc, 'Val Chart Type Acc', str(chart_type))

def checkpoint_stuff(model, optimizer, epoch, args, model_args, iter_idx=0,
                     **kwargs):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # model
    model_path = pth.join(args.checkpoint_dir, 'model_ep{}.pt'.format(epoch))
    torch.save({
        'model_args': model_args,
        'state_dict': model.state_dict(),
    }, model_path)
    # optimizer
    optim_path = pth.join(args.checkpoint_dir, 'optim_ep{}.pt'.format(epoch))
    torch.save({
        'optimizer': optimizer,
        'iter_idx': iter_idx,
        'epoch': epoch,
    }, optim_path)

def main(args):
    global running_loss, start_t
    # logging info that needs to persist across iterations
    viz = utils.visualize.VisdomVisualize(env_name=args.env_name)
    viz.viz.text(str(args))
    running_loss = None
    running_accs = {qtype: 0.5 for qtype, _ in enumerate(utils.QTYPE_ID_TO_META)}
    start_t = None

    # data
    dataset = FigQADataset(args.figqa_dir, args.figqa_pre,
                           split='train1')
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=True,
                            shuffle=bool(args.shuffle_train))
    val_dataset = FigQADataset(args.figqa_dir, args.figqa_pre,
                               split=args.val_split)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True,
                                shuffle=True)

    # model
    if args.start_from:
        model = utils.load_model(fname=args.start_from, ngpus=args.cuda)
    else:
        model_args = figqa.options.model_args(args)
        model = utils.load_model(model_args, ngpus=args.cuda)

    # optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    def exp_lr(epoch):
        iters = epoch * len(dataloader)
        return args.lr_decay**iters
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, exp_lr)
    criterion = nn.NLLLoss()

    # training
    for epoch in range(args.epochs):
        checkpoint_stuff(**locals())
        scheduler.step()
        start_t = timer()
        # TODO: understand when/why automatic garbage collection slows down
        # the train loop
        gc.disable()
        data_time = time.time()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        for local_iter_idx, batch in batch_iter(dataloader, args):
            start.record()
            torch.cuda.synchronize()

            iter_idx = local_iter_idx + epoch * len(dataloader)

            # forward + update
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch['answer'])
            loss.backward()
            optimizer.step()

            #Chargrid: Gradient Clipping
            #torch.nn.utils.clip_grad_value_([model.module.decision_weight],0.1)
            

            # visualize, log, checkpoint
            #print("visualize")
            log_stuff(**locals())
            #for idx,timing in enumerate(["FW","Loss","Backw","Opt","log"]):
                #print(timing,times[idx+1]-times[idx])
            #data_time = time.time()
            end.record()
            torch.cuda.synchronize()
            #print("Training: ",start.elapsed_time(end))
        gc.enable()
        


if __name__ == '__main__':
    def char_split(document):
        return list(document.lower())

    #os.system("taskset -p 0xff %d" % os.getpid())
    
    main(figqa.options.parse_arguments())
