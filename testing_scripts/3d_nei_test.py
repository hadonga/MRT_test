import numpy as np
import matplotlib.pyplot as plt
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics
import unfoldNd
from scipy import ndimage as sci

def get_gaussian_kernel_3D(kernel_size=3, sigma=2):
  x_coord = torch.arange(kernel_size)
  x_grid = x_coord.repeat(kernel_size,kernel_size).view(kernel_size,kernel_size, kernel_size)
  slice=x_grid[0].t()
  y_grid = slice.repeat(kernel_size,1,1)
  z_grid = torch.zeros((kernel_size,kernel_size,kernel_size),dtype=torch.int64)
  for i in range (kernel_size):
    z_grid[i]=i

  xyz_grid = torch.stack([x_grid, y_grid,z_grid], dim=-1).float()
  mean = (kernel_size - 1) / 2.
  variance = sigma**2.
  gaussian_kernel = (1. / ((2. * math.pi)**(0.5) * sigma)**3) *torch.exp(-torch.sum((xyz_grid - mean)**2., dim=-1) / (2 * variance))
  gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
  gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size,kernel_size)

  return gaussian_kernel



k_labels=np.load("nearest_labels.npy")
k_ranges=np.load("nearest_range.npy")
ori_range=np.load("saved/frame75/unproj_range.npy")

search=5
sigma=1.0
knn=5
cutoff=1.0

k_labels=torch.from_numpy(k_labels)
ori_range=torch.from_numpy(ori_range)

center = int(((search * search *search) - 1) / 2)

k_ranges[center, :] = ori_range
k2_distances = torch.abs(torch.from_numpy(k_ranges) - ori_range)

inv_gauss_k = (1 - get_gaussian_kernel_3D(search, sigma)).view(-1, 1)
# inv_gauss_k = inv_gauss_k.to(device).type(proj_range.type())

# apply weighing
k2_distances = k2_distances * inv_gauss_k

knn_distances, knn_idx = k2_distances.topk(knn, dim=0, largest=False, sorted=False)

knn_distances=knn_distances.numpy()
# knn_idx=knn_idx.numpy()


knn_argmax = torch.gather(input=k_labels, dim=0, index=knn_idx)

# knn_argmax=knn_argmax.numpy()

nclasses=21

if cutoff > 0:
  knn_distances = torch.gather(input=k2_distances, dim=1, index=knn_idx)
  knn_invalid_idx = knn_distances > cutoff
  knn_argmax[knn_invalid_idx] = nclasses

knn_argmax_onehot = torch.zeros((1, nclasses + 1, 100000)).type(torch.float64)
# knn_argmax=torch.from_numpy(knn_argmax)

knn_argmax= knn_argmax.type(torch.int64)
knn_argmax =knn_argmax[None,...]
ones = torch.ones_like(knn_argmax).type(torch.float64)

knn_argmax_onehot = knn_argmax_onehot.scatter_add_(1, knn_argmax, ones)

knn_argmax_out = knn_argmax_onehot[:, 1:-1].argmax(dim=1) + 1
knn_argmax_out = knn_argmax_out.view(100000)

knn_argmax_out=knn_argmax_out.numpy()

ground_truth = np.load("saved/frame75/curr_lab.npy")

f1 = sklearn.metrics.f1_score(ground_truth, knn_argmax_out, average='micro')
iou = sklearn.metrics.jaccard_score(ground_truth, knn_argmax_out, average='micro')

# kernal_3D=get_gaussian_kernel_3D(5,1.0)
# kernal_2D=kernal_3D.numpy().reshape(125,1)
print("test")