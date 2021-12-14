# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataset_kitti import Dataset_kitti
from unet import UNet
from torch.utils.data import DataLoader
# from module.network import UNet
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torchsummary import summary
import wandb
import yaml
from original_modules.segmentator import *

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from original_modules.warmupLR import *
from original_modules.ioueval import *
from original_modules.avgmeter import *
from original_modules.sync_batchnorm.batchnorm import *

import time
from tqdm.notebook import tqdm
from torchsummary import summary
import wandb
from argparse import ArgumentParser


def get_param():

    # loading arch and kitti files
    archfile = 'config/arch/initial_arch.yaml'
    configfile = 'config/labels/semantickitti/semantic-kitti-color-mod.yaml'

    # print(torch.__version__)
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--dataset_dir', default='/home/share/dataset/semanticKITTI/sequences/')
    # parser.add_argument('--dataset_dir', default='/root/dataset/kitti/sequences/')
    parser.add_argument('--log_dir', default='lightning_logs')
    parser.add_argument('--n_channels', type=int, default=5)
    parser.add_argument('--n_classes', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=18)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)

    hparams = parser.parse_args()

    with open(archfile, 'r') as stream:
        kitti_param = yaml.safe_load(stream)
    with open(configfile, 'r') as stream:
        kitti_config = yaml.safe_load(stream)

    return hparams, kitti_param, kitti_config

def map_to_xentropy(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]



def loss_settings(epsilon_w = 0.001):

    content = torch.zeros(n_classes, dtype=torch.float)
    learning_map = kitti_config["learning_map"]
    items =DATA["content"].items()
    for cl, freq in DATA["content"].items():
        x_cl = map_to_xentropy(cl,kitti_config["learning_map"])  # map actual class to xentropy class
        print("class : ",x_cl,"freq",freq)
        content[x_cl] += freq

    loss_w = 1 / (content + epsilon_w)  # get weights

    for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
        if DATA["learning_ignore"][x_cl]:
            # don't weigh
            loss_w[x_cl] = 0

    # print("Loss weights from content: ", loss_w.data)


    return  loss_w


########################################################################


def train_epoch(train_loader, model, criterion, optimizer, epoch, evaluator, scheduler, report=10, show_scans=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    iou = AverageMeter()
    update_ratio_meter = AverageMeter()

    # empty the cache to train now
    if gpu:
      torch.cuda.empty_cache()

    # switch to train mode
    model.train()

    end = time.time()

    # for i, (in_vol, proj_labels, _, _, _, _, _, _,proj_mask) in enumerate(train_loader):
    for i, data in enumerate(dataloader):
        in_vol, proj_labels,proj_mask = data["proj_input_total_time_t"], data["proj_single_label"], data["proj_mask"]

        # measure data loading time
        data_time.update(time.time() - end)
        if not multi_gpu and gpu:
            in_vol = in_vol.cuda()
            proj_mask = proj_mask.cuda()
        if gpu:
            proj_labels = proj_labels.cuda(non_blocking=True).long()

        # compute output
        output = model(in_vol, proj_mask)
        # print(output.shape)
        loss = criterion(torch.log(output.clamp(min=1e-8)), proj_labels)
        # loss = loss_crs(output,proj_labels)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if n_gpus > 1:
            idx = torch.ones(n_gpus).cuda()
            loss.backward(idx)
        else:
            loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        loss = loss.mean()
        with torch.no_grad():
            evaluator.reset()
            argmax = output.argmax(dim=1)
            # argmax_numpy=argmax.cpu().numpy()
            evaluator.addBatch(argmax, proj_labels)
            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
        losses.update(loss.item(), in_vol.size(0))
        acc.update(accuracy.item(), in_vol.size(0))
        iou.update(jaccard.item(), in_vol.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # get gradient updates and weights, so I can print the relationship of
        # their norms
        update_ratios = []
        for g in optimizer.param_groups:
            lr = g["lr"]
            for value in g["params"]:
                if value.grad is not None:
                    w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                    update = np.linalg.norm(-max(lr, 1e-10) *
                                            value.grad.cpu().numpy().reshape((-1)))
                    update_ratios.append(update / max(w, 1e-10))
        update_ratios = np.array(update_ratios)
        update_mean = update_ratios.mean()
        update_std = update_ratios.std()
        update_ratio_meter.update(update_mean)  # over the epoch


        if i % ARCH["train"]["report_batch"] == 0:
            print('Lr: {lr:.3e} | '
                  'Update: {umean:.3e} mean,{ustd:.3e} std | '
                  'Epoch: [{0}][{1}/{2}] | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                  'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=acc, iou=iou, lr=lr,
                umean=update_mean, ustd=update_std))

        # step scheduler
        scheduler.step()
    return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg


##########################################################################

hparams, kitti_param, kitti_config = get_param()

data_dir='/home/share/dataset/semanticKITTI/sequences/'

batchSize=4

train_dataset = Dataset_kitti(data_dir,kitti_param, kitti_config, step='train')

# data=train_dataset[0]
# input,gt=data["proj_input_total_time_t"], data["proj_single_label"]
# show_2D_range(gt, "ground truth", 40)
# show_2D_image(input[1], "input", 40)

dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

ARCH = yaml.safe_load(open('config/darknet53-1024px.yaml', 'r'))
DATA = yaml.safe_load(open('config/semantic-kitti-all.yaml', 'r'))
########################################################
# Testing
# dataset = kitti_loader()
# projection,label =dataset.__getitem__(index=0)
# print(np.shape(projection),np.shape(label))
# print("stop")

# experiment = wandb.init(project='RangeNet_test',entity='furqanabid' ,resume='allow')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


info = {"train_update": 0,
                 "train_loss": 0,
                 "train_acc": 0,
                 "train_iou": 0,
                 "valid_loss": 0,
                 "valid_acc": 0,
                 "valid_iou": 0,
                 "backbone_lr": 0,
                 "decoder_lr": 0,
                 "head_lr": 0,
                 "post_lr": 0}



n_classes=hparams.n_classes


# loss_crs = nn.CrossEntropyLoss(reduction='mean').cuda()

loss_w=loss_settings(0.001)



with torch.no_grad():
    model = Segmentator(ARCH, n_classes)


# GPU?
gpu = False
multi_gpu = False
n_gpus = 0
model_single = model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training in device: ", device)
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
  cudnn.benchmark = True
  cudnn.fastest = True
  gpu = True
  n_gpus = 1
  model.cuda()
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)   # spread in gpus
  model = convert_model(model).cuda()  # sync batchnorm
  model_single = model.module  # single model to get weight names
  multi_gpu = True
  n_gpus = torch.cuda.device_count()

# loss
if "loss" in ARCH["train"].keys() and ARCH["train"]["loss"] == "xentropy":
  criterion = nn.NLLLoss(weight=loss_w).to(device)
else:
  raise Exception('Loss not defined in config file')
# loss as dataparallel too (more images in batch)
if n_gpus > 1:
  criterion = nn.DataParallel(criterion).cuda()  # spread in gpus

# optimizer
if ARCH["post"]["CRF"]["use"] and ARCH["post"]["CRF"]["train"]:
  lr_group_names = ["post_lr"]
  train_dicts = [{'params': model_single.CRF.parameters()}]
else:
  lr_group_names = []
  train_dicts = []
if ARCH["backbone"]["train"]:
  lr_group_names.append("backbone_lr")
  train_dicts.append(
      {'params': model_single.backbone.parameters()})
if ARCH["decoder"]["train"]:
  lr_group_names.append("decoder_lr")
  train_dicts.append(
      {'params': model_single.decoder.parameters()})
if ARCH["head"]["train"]:
  lr_group_names.append("head_lr")
  train_dicts.append({'params': model_single.head.parameters()})

# Use SGD optimizer to train
optimizer = torch.optim.SGD(train_dicts,lr=ARCH["train"]["lr"],momentum=ARCH["train"]["momentum"],weight_decay=ARCH["train"]["w_decay"])

# Use warmup learning rate
# post decay and step sizes come in epochs and we want it in steps
steps_per_epoch = len(dataloader)
up_steps = int(ARCH["train"]["wup_epochs"] * steps_per_epoch)
final_decay = ARCH["train"]["lr_decay"] ** (1/steps_per_epoch)
scheduler = warmupLR(optimizer=optimizer,lr=ARCH["train"]["lr"],warmup_steps=up_steps,momentum=ARCH["train"]["momentum"],decay=final_decay)


#################################################################################################################

# accuracy and IoU stuff
best_train_iou = 0.0
best_val_iou = 0.0

ignore_class = []
# for i, w in enumerate(loss_w):
#     if w < 1e-10:
#         ignore_class.append(i)
#         print("Ignoring class ", i, " in IoU evaluation")
evaluator = iouEval(n_classes,device,ignore_class)

# train for n epochs
for epoch in range(ARCH["train"]["max_epochs"]):
# get info for learn rate currently
    groups = optimizer.param_groups
    for name, g in zip(lr_group_names, groups):
        info[name] = g['lr']

    # train for 1 epoch
    acc, iou, loss, update_mean = train_epoch(train_loader=dataloader, model=model, criterion=criterion,
                                                 optimizer=optimizer, epoch=epoch, evaluator=evaluator,
                                                 scheduler=scheduler,report=ARCH["train"]["report_batch"],show_scans=ARCH["train"]["show_scans"])

    # update info
    info["train_update"] = update_mean
    info["train_loss"] = loss
    info["train_acc"] = acc
    info["train_iou"] = iou

    # logdir='original_modules\saved_models'
    # remember best iou and save checkpoint
    if iou > best_train_iou:
        print("Best mean iou in training set so far, save model!")
        best_train_iou = iou
        model_single.save_checkpoint(suffix="_train")
    # # update info
    # info["valid_loss"] = loss
    # info["valid_acc"] = acc
    # info["valid_iou"] = iou
    #
    # # remember best iou and save checkpoint
    # if iou > best_val_iou:
    #   print("Best mean iou in validation so far, save model!")
    #   print("*" * 80)
    #   best_val_iou = iou
    #
    #   # save the weights!
    # model_single.save_checkpoint(suffix="")

    print("*" * 80)

print('Finished Training')

