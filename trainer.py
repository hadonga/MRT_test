from dataset_kitti import Dataset_kitti
import yaml
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm.notebook import tqdm
from torchsummary import summary
import wandb
from argparse import ArgumentParser


proj_W=1024
proj_H=64



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


def show_2D_range(range_2D, title, scale, show_plot=True, save_image=False, path=''):
    colormapfile='config/labels/semantickitti/colormap.yaml'
    with open(colormapfile, 'r') as stream:
         colormapfile = yaml.safe_load(stream)
    color_map=colormapfile['color_map']
    plt.figure(figsize=((proj_W / scale) - 2, (proj_H / scale) + 0.5))
    plt.title(title)
    image = np.array([[color_map[val] for val in row] for row in range_2D], dtype='B')
    plt.imshow(image)
    plt.show()

    # 2D visualization any image

def show_2D_image(image, title, scale, colormap="magma"):
    cmap = plt.cm.get_cmap(colormap, 10)
    plt.figure(figsize=((proj_W / scale) - 2, (proj_H / scale) + 0.5))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    # plt.colorbar()
    plt.show()




hparams, kitti_param, kitti_config = get_param()

data_dir='/home/share/dataset/semanticKITTI/sequences/'

batchSize=4

train_dataset = Dataset_kitti(data_dir,kitti_param, kitti_config, step='train')

# data=train_dataset[0]
# input,gt=data["proj_input_total_time_t"], data["proj_single_label"]
# show_2D_range(gt, "ground truth", 40)
# show_2D_image(input[1], "input", 40)

dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

# datadict = train_dataset[75]
#
# data=datadict["data"]
# gt=datadict["gt_multi_pixel"]


# wandb.init()

model= UNet(hparams.n_channels, hparams.n_classes)
print("Model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
#
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.cuda()

loss_crs = nn.CrossEntropyLoss(ignore_index=hparams.n_classes,reduction='mean').cuda()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.005,momentum=0.9, weight_decay=0.0001)
#
#
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.35, patience=5, verbose=True,
#                                                        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
#                                                        eps=1e-08)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True,
                                                       threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                       eps=1e-08)

data_length=len(train_dataset)

summary(model, input_size=(5, 64, 1024))
# print(model)

step_losses = []
epoch_losses = []

def main():
    epochs = 25
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0

        for batch_idx, data in enumerate(dataloader):
            input,gt=data["single_data"], data["gt_single_pixel"]

            # print(proj_remission.shape)
            # print("Epoch == ", epoch, "Batch == ", batch_idx)
            # proj_remission, proj_sem_label = proj_remission.cuda(), proj_sem_label.cuda()
            input = input.cuda()
            # proj_sem_label = torch.tensor(proj_sem_label, dtype=torch.long, device=torch.device('cuda'))
            gt = gt.cuda(non_blocking=True).long()
            # print("proj_sem_label shape", proj_sem_label.size())
            optimizer.zero_grad()
            out = model(input)
            loss = loss_crs(out, gt)
            print("Batch", batch_idx," of ",int(data_length/batchSize), "epoch", epoch,"loss",loss.item())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())
            # wandb.log({'loss': loss.item()})
        epoch_losses.append(epoch_loss / len(dataloader))
        # wandb.log({'epoch_loss': epoch_losses[batch_idx]})
        print("epoch loss",epoch_losses)



if __name__ == '__main__':
    main()

    model_name = "test_model.pth"
    torch.save(model.state_dict(), model_name)

    print("done")


