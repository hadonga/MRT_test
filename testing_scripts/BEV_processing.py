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


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
  # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
  x_coord = torch.arange(kernel_size)
  x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
  y_grid = x_grid.t()
  xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
  test = xy_grid.numpy()

  mean = (kernel_size - 1) / 2.
  variance = sigma**2.

  # Calculate the 2-dimensional gaussian kernel which is
  # the product of two gaussian distributions for two different
  # variables (in this case called x and y)
  gaussian_kernel = (1. / (2. * math.pi * variance)) *\
      torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2 * variance))

  # Make sure sum of values in gaussian kernel equals 1.
  gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

  # Reshape to 2d depthwise convolutional weight
  gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)

  return gaussian_kernel

def gen_gaussian_kernel(shape, mean, var):
    coors = [range(shape[d]) for d in range(len(shape))]
    k = np.zeros(shape=shape)
    cartesian_product = [[]]
    for coor in coors:
        cartesian_product = [x + [y] for x in cartesian_product for y in coor]
    for c in cartesian_product:
        s = 0
        for cc, m in zip(c,mean):
            s += (cc - m)**2
        k[tuple(c)] = np.exp(-s/(2*var))
    return k



class KNN(nn.Module):
    def __init__(self, knn=5, search=5, sigma=1.0,
                 cutoff=1.0, nclasses=21):
        super().__init__()
        self.knn = knn
        self.search = search
        self.sigma = sigma
        self.cutoff = cutoff
        self.nclasses = nclasses
        print("kNN parameters:")
        print("knn:", self.knn)
        print("search:", self.search)
        print("sigma:", self.sigma)
        print("cutoff:", self.cutoff)
        print("nclasses:", self.nclasses)

    def forward(self, proj_range, unproj_range, proj_argmax, px, py):
        ''' Warning! Only works for un-batched pointclouds.
            If they come batched we need to iterate over the batch dimension or do
            something REALLY smart to handle unaligned number of points in memory
        '''
        # get device
        if proj_range.is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # sizes of projection scan
        H, W = proj_range.shape

        # number of points
        P = unproj_range.shape

        # check if size of kernel is odd and complain
        if (self.search % 2 == 0):
            raise ValueError("Nearest neighbor kernel must be odd number")

        # calculate padding
        pad = int((self.search - 1) / 2)

        # unfold neighborhood to get nearest neighbors for each pixel (range image)
        proj_unfold_k_rang = F.unfold(proj_range[None, None, ...],
                                      kernel_size=(self.search, self.search),
                                      padding=(pad, pad))

        # index with px, py to get ALL the pcld points
        idx_list = py * W + px
        unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list]

        # WARNING, THIS IS A HACK
        # Make non valid (<0) range points extremely big so that there is no screwing
        # up the nn self.search
        unproj_unfold_k_rang[unproj_unfold_k_rang < 0] = float("inf")

        # now the matrix is unfolded TOTALLY, replace the middle points with the actual range points
        center = int(((self.search * self.search) - 1) / 2)
        unproj_unfold_k_rang[:, center, :] = unproj_range

        # now compare range
        k2_distances = torch.abs(unproj_unfold_k_rang - unproj_range)

        # make a kernel to weigh the ranges according to distance in (x,y)
        # I make this 1 - kernel because I want distances that are close in (x,y)
        # to matter more



        inv_gauss_k = (
                1 - get_gaussian_kernel(self.search, self.sigma, 1)).view(1, -1, 1)
        inv_gauss_k = inv_gauss_k.to(device).type(proj_range.type())

        # apply weighing
        k2_distances = k2_distances * inv_gauss_k

        # find nearest neighbors
        _, knn_idx = k2_distances.topk(
            self.knn, dim=1, largest=False, sorted=False)

        # do the same unfolding with the argmax
        proj_unfold_1_argmax = F.unfold(proj_argmax[None, None, ...].float(),
                                        kernel_size=(self.search, self.search),
                                        padding=(pad, pad)).long()
        unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list]

        # get the top k predictions from the knn at each pixel
        knn_argmax = torch.gather(
            input=unproj_unfold_1_argmax, dim=1, index=knn_idx)

        # fake an invalid argmax of classes + 1 for all cutoff items
        if self.cutoff > 0:
            knn_distances = torch.gather(input=k2_distances, dim=1, index=knn_idx)
            knn_invalid_idx = knn_distances > self.cutoff
            knn_argmax[knn_invalid_idx] = self.nclasses

        # now vote
        # argmax onehot has an extra class for objects after cutoff
        knn_argmax_onehot = torch.zeros(
            (1, self.nclasses + 1, P[0]), device=device).type(proj_range.type())
        ones = torch.ones_like(knn_argmax).type(proj_range.type())
        knn_argmax_onehot = knn_argmax_onehot.scatter_add_(1, knn_argmax, ones)

        # now vote (as a sum over the onehot shit)  (don't let it choose unlabeled OR invalid)
        knn_argmax_out = knn_argmax_onehot[:, 1:-1].argmax(dim=1) + 1

        # reshape again
        knn_argmax_out = knn_argmax_out.view(P)

        return knn_argmax_out



class BEV_KNN(nn.Module):
    def __init__(self, knn=5, search=3, sigma=1.0,
                 cutoff=1.0, nclasses=21):
        super().__init__()
        self.knn = knn
        self.search = search
        self.sigma = sigma
        self.cutoff = cutoff
        self.nclasses = nclasses
        print("kNN parameters:")
        print("knn:", self.knn)
        print("search:", self.search)
        print("sigma:", self.sigma)
        print("cutoff:", self.cutoff)
        print("nclasses:", self.nclasses)

    def forward(self, proj_range, unproj_range, proj_argmax, px, py, pz):
        if proj_range.is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        D, H, W = proj_range.shape
        P = unproj_range.shape

        pad = int((self.search - 1) / 2)

        proj_unfold_k_rang = unfoldNd.unfoldNd(proj_range[None,None, ...],kernel_size=(self.search,self.search,self.search),
                                              dilation=(1,1,1),padding=(pad,pad,pad),stride=(1,1,1))

        idx_list = px + W *(py + H * pz)

        unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list]

        unproj_unfold_k_rang[unproj_unfold_k_rang < 0] = float("inf")

        center = int(((self.search * self.search *self.search) - 1) / 2)
        unproj_unfold_k_rang[:, center, :] = unproj_range
        k2_distances = torch.abs(unproj_unfold_k_rang - unproj_range)

        gauss_kernal_input = np.full((self.search, self.search, self.search), 1, dtype=np.float64)
        gauss_kernal_input = sci.gaussian_filter(gauss_kernal_input, sigma=self.sigma)


        gauss_3d = gen_gaussian_kernel((self.search,self.search,self.search), (pad,pad,pad), self.sigma)
        inv_gauss_k = (1.2 - torch.from_numpy(gauss_3d)).view(1, -1, 1)


        # gauss_kernal=1 - get_gaussian_kernel(self.search, self.sigma, 1)
        # gauss_kernal=gauss_kernal.numpy()
        #
        # # to matter more
        # inv_gauss_k = (1 - get_gaussian_kernel(self.search, self.sigma, 1)).view(1, -1, 1)
        inv_gauss_k = inv_gauss_k.to(device).type(proj_range.type())
        # k2_distances = k2_distances * inv_gauss_k
        _, knn_idx = k2_distances.topk(self.knn, dim=1, largest=False, sorted=False)

        proj_unfold_1_argmax = unfoldNd.unfoldNd(proj_argmax[None, None, ...],
                                               kernel_size=(self.search, self.search, self.search),
                                               dilation=(1, 1, 1), padding=(pad, pad, pad), stride=(1, 1, 1))

        unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list]

        knn_argmax = torch.gather(input=unproj_unfold_1_argmax, dim=1, index=knn_idx).type(torch.int64)

        if self.cutoff > 0:
            knn_distances = torch.gather(input=k2_distances, dim=1, index=knn_idx)
            knn_invalid_idx = knn_distances > self.cutoff
            knn_argmax[knn_invalid_idx] = self.nclasses


        knn_argmax_onehot = torch.zeros((1, self.nclasses + 1, P[0]), device=device).type(proj_range.type())
        ones = torch.ones_like(knn_argmax).type(proj_range.type())

        knn_argmax_onehot = knn_argmax_onehot.scatter_add_(1, knn_argmax, ones)


        knn_argmax_out = knn_argmax_onehot[:, 1:-1].argmax(dim=1) + 1
        knn_argmax_out = knn_argmax_out.view(P)

        return knn_argmax_out


def BEV_projection(xyz,labels,resolution):

    x = xyz[:, 0]
    y = xyz[:, 1]
    # z = xyz[:, 2]

    xlimit = [-80, 80]
    ylimit = [-16, 32]
    # zlimit = [-3.5, 3]

    x = x - xlimit[0]
    y = y - ylimit[0]

    x = x / (xlimit[1] - xlimit[0])
    y = y / (ylimit[1] - ylimit[0])

    f = int(1/resolution)
    W = (xlimit[1] + abs(xlimit[0])) * f
    H = (ylimit[1] + abs(ylimit[0])) * f

    x = np.floor(x * W).astype(np.int32)
    y = np.floor(y * H).astype(np.int32)

    depth = np.linalg.norm(xyz, 2, axis=1)

    proj_x = np.minimum(W - 1, x)
    proj_x = np.maximum(0, proj_x)

    proj_y = np.minimum(W - 1, y)
    proj_y = np.maximum(0, proj_y)

    BEV_labels = np.full((H, W), -1, dtype=np.int32)
    BEV_labels[proj_y, proj_x] = labels

    BEV_range = np.full((H, W), -1, dtype=np.float64)
    BEV_range[proj_y, proj_x] = depth

    return BEV_labels,BEV_range,proj_x,proj_y

def voxelized_pointcloud(xyz,labels,xyz_resolution):

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    xlimit = [-80, 80]
    ylimit = [-16, 32]
    zlimit = [-3.5, 3]

    x = x - xlimit[0]
    y = y - ylimit[0]
    z = z - zlimit[0]

    x = x / (xlimit[1] - xlimit[0])
    y = y / (ylimit[1] - ylimit[0])
    z = z / (zlimit[1] - zlimit[0])

    W = (xlimit[1] + abs(xlimit[0])) * int(1/xyz_resolution[0])
    H = (ylimit[1] + abs(ylimit[0])) * int(1/xyz_resolution[1])
    # D = int((zlimit[1] + abs(zlimit[0])) * int(1/xyz_resolution[2]))
    D=xyz_resolution[2]

    x = np.floor(x * W).astype(np.int32)
    y = np.floor(y * H).astype(np.int32)
    z = np.floor(z * D).astype(np.int32)

    depth = np.linalg.norm(xyz, 2, axis=1)

    proj_x = np.minimum(W - 1, x)
    proj_x = np.maximum(0, proj_x)

    proj_y = np.minimum(W - 1, y)
    proj_y = np.maximum(0, proj_y)

    proj_z = np.minimum(D-1 , z)
    proj_z = np.maximum(0, proj_z)

    BEV_labels = np.full((D, H, W), -1, dtype=np.int32)
    BEV_labels[proj_z, proj_y, proj_x] = labels

    BEV_range = np.full((D, H, W), -1, dtype=np.float64)
    BEV_range[proj_z, proj_y, proj_x] = depth

    return BEV_labels,BEV_range,proj_x,proj_y,proj_z

def show_2D_range(range_2D,W,H,color_map):
    plt.figure(figsize=((W/40) - 2, (H/40) + 0.5))
    # plt.title(title)
    image = np.array([[color_map[val] for val in row] for row in range_2D], dtype='B')
    plt.imshow(image)
    plt.show()


def knn_check():
    archfile = 'config/arch/initial_arch.yaml'

    with open(archfile, 'r') as stream:
        param = yaml.safe_load(stream)

    proj_range = np.load("saved/frame75/projected_curr_range.npy")
    unproj_range = np.load("saved/frame75/unproj_range.npy")
    proj_argmax = np.load("saved/frame75/labels.npy")
    px = np.load("saved/frame75/p_x.npy")
    py = np.load("saved/frame75/p_y.npy")

    W=1024
    idx_list = py * W + px
    proj_argmax=proj_argmax.reshape(64*1024,-1)
    pred=proj_argmax[idx_list]

    ground_truth = np.load("saved/frame75/curr_lab.npy")

    f1 = sklearn.metrics.f1_score(ground_truth, pred, average='micro')
    iou = sklearn.metrics.jaccard_score(ground_truth, pred, average='micro')

    knn_process = KNN()

    unproj_argmax = knn_process(torch.from_numpy(proj_range), torch.from_numpy(unproj_range)
                                , torch.from_numpy(proj_argmax), px, py)

    predicted_res_knn = unproj_argmax.numpy()

    f1=sklearn.metrics.f1_score(ground_truth,predicted_res_knn,average='micro')
    iou=sklearn.metrics.jaccard_score(ground_truth,predicted_res_knn,average='micro')

    print("stopped")


def show_2D_image( image,W,H, cmap="tab20c"):
    # cmap = plt.cm.get_cmap(cmap, 10)
    plt.figure(figsize=((W/40) - 2, (H/40) + 0.5))
    plt.imshow(image,cmap="tab20c")
    # plt.clim(0, 80)
    plt.colorbar()
    plt.show()

def bev_check():
    xyz = np.load("saved/frame75/curr_points.npy")
    ground_truth=np.load("saved/frame75/curr_lab.npy")
    proj_range = np.load("saved/frame75/projected_curr_range.npy")
    unproj_range = np.load("saved/frame75/unproj_range.npy")
    proj_argmax = np.load("saved/frame75/labels.npy")
    px = np.load("saved/frame75/p_x.npy")
    py = np.load("saved/frame75/p_y.npy")

    idx_list = py * 1024 + px
    proj_argmax = proj_argmax.reshape(64 * 1024, -1)
    pred = proj_argmax[idx_list]

    pred=np.squeeze(pred)

    depth = np.linalg.norm(xyz, 2, axis=1)
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]



    xyz_resolution=[1/5,1/5,32]
    voxel_labels,voxel_range,proj_x,proj_y,proj_z=voxelized_pointcloud(xyz, pred,xyz_resolution)



    # prev_x,next_x=proj_x-1,proj_x+1
    # x=np.hstack((np.expand_dims(prev_x,axis=1),np.expand_dims(proj_x,axis=1),np.expand_dims(next_x,axis=1)))
    # prev2_x, next2_x = proj_x - 2, proj_x + 2
    # x = np.hstack((np.expand_dims(prev2_x, axis=1), x, np.expand_dims(next2_x, axis=1)))
    #
    # prev_y, next_y = proj_y - 1, proj_y + 1
    # y = np.hstack((np.expand_dims(prev_y, axis=1), np.expand_dims(proj_y, axis=1), np.expand_dims(next_y, axis=1)))
    # prev2_y, next2_y = proj_y - 2, proj_y + 2
    # y = np.hstack((np.expand_dims(prev2_y, axis=1), y, np.expand_dims(next2_y, axis=1)))
    #
    # prev_z, next_z = proj_z - 1, proj_z + 1
    # z = np.hstack((np.expand_dims(prev_z, axis=1), np.expand_dims(proj_z, axis=1), np.expand_dims(next_z, axis=1)))
    # prev2_z, next2_z = proj_z - 2, proj_z + 2
    # z = np.hstack((np.expand_dims(prev2_z, axis=1), z, np.expand_dims(next2_z, axis=1)))
    #
    # nearest_labels = np.full((125,100000),-1,dtype=np.int32)
    # nearest_range = np.full((125,100000),-1,dtype=np.float32)
    #
    #
    # for i in range (100000):
    #     n=0
    #     for depth in range (5):
    #         for height in range (5):
    #             for width in range (5):
    #                 if z[i,depth] >= 0 and z[i,depth] < 32 :
    #                     nearest_labels[n,i]=voxel_labels[z[i,depth],y[i,height],x[i,width]]
    #                     nearest_range[n,i]=voxel_range[z[i,depth],y[i,height],x[i,width]]
    #                 else :
    #                     nearest_labels[n, i]=-1
    #                     nearest_range[n, i]=-1
    #                 n+=1
    #
    # np.save("nearest_labels.npy",nearest_labels)
    # np.save("nearest_range.npy", nearest_range)

    v1,v3=np.unique(voxel_labels,False,False,True)

    a,b,c= np.shape(voxel_labels)

    percentage = (v3[0]*100)/(a*b*c)

    #only testing stats part
    # xyz_proj = np.vstack((proj_x, proj_y,proj_z))
    # unique, indices, count = np.unique(xyz_proj, return_index=True, return_counts=True, axis=1)
    # one_to_one = count[count==1]
    # more_than_1 = count[count > 1]

    # proj_argmax,proj_range,px,py=BEV_projection(xyz, pred, 1/10)
    # knn_process = KNN()
    # unproj_argmax = knn_process(torch.from_numpy(proj_range), torch.from_numpy(unproj_range)
    #                             , torch.from_numpy(proj_argmax), px, py)

    knn_process1 = BEV_KNN()
    unproj_argmax = knn_process1(torch.from_numpy(voxel_range), torch.from_numpy(unproj_range)
                                , torch.from_numpy(voxel_labels), proj_x, proj_y,proj_z)

    predicted_res_knn = unproj_argmax.numpy()

    f1 = sklearn.metrics.f1_score(ground_truth, predicted_res_knn, average='micro')
    iou = sklearn.metrics.jaccard_score(ground_truth, predicted_res_knn, average='micro')

    # W, H = 1600, 480
    #
    # show_2D_image(BEV_range, W, H)


    configfile = 'config/labels/semantickitti/colormap.yaml'
    with open(configfile, 'r') as stream:
        kitti_config = yaml.safe_load(stream)
    color_map = kitti_config['color_map']

    # show_2D_range(BEV_labels, W, H, color_map)

    xy=np.vstack((proj_x,proj_y))
    unique,indices,count = np.unique(xy,return_index=True,return_counts=True,axis=1)

    more_than_1=count[count==1]

    xy_range = np.vstack((px, py))
    unique_range, indices_range, count_range = np.unique(xy_range, return_index=True, return_counts=True, axis=1)

    more_than_1_range = count_range[count_range==1]

    print("1")

if __name__ == '__main__':
    # knn_check()
    bev_check()