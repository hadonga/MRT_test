import time

import numba
import numpy as np

# Source code from pillarpoints
# Thanks for sharing.

@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=100,
                            max_voxels=5000):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size  # [-51.2, -51.2, -4, 51.2, 51.2, 4] / [0.8,0.8,8] ->[128,128,1]
    grid_size = np.round(grid_size, 0,grid_size).astype(np.int32) #[128,128,1] 在numba中的写法不一样。

    # lower_bound = coors_range[:3]
    # upper_bound = coors_range[3:]

    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    # failed = False
    for i in range(N):
        failed = False
        for j in range(ndim): # (X,Y,Z) -> 128,128,1
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j]) # voxel_size=[0.8,0.8,8]
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue

        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]] # coor_to_voxelidx 128,128,1 [-1,-1,-1, ..., -1]
        if voxelidx == -1: # new points in a pillar
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1 # voxel_num=1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx # -1 -> 0
            coors[voxelidx] = coor # coors[0]= coor
        num = num_points_per_voxel[voxelidx] # 0
        if num < max_points: # max_points:35
            voxels[voxelidx, num] = points[i] # [0,0] <- i
            num_points_per_voxel[voxelidx] += 1 #
    # print("===in_numba")
    # print(voxel_num)
    # print(voxels.shape)
    # print("in_numba===")
    return voxel_num

def points_to_voxel(points, # input
                     voxel_size, # [0.8, 0.8, 8]
                     coors_range, #[-51.2, -51.2, -4, 51.2, 51.2, 4]
                     max_points= 100,
                     max_voxels=5000): # total number of voxels: 128*128= 16384
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud)
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax  x_min，x_max,y_min ...
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points. # 三个M是一样？ 每个voxel里面的点？
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)

    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())

    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32) #5000*1
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32) #128*128*1 [-1,-1,-1...-1]
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype) # 5000*100*4
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32) #5000*3

    voxel_num = _points_to_voxel_kernel(  # return 有点的柱子数
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num] # voxel_num x3
    voxels = voxels[:voxel_num] #voxel_num x 100 x 4
    num_points_per_voxel = num_points_per_voxel[:voxel_num] #voxel_numx1

    # voxels[:, :, -3:] = voxels[:, :, :3] - \
    #     voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)
    return voxels, coors, num_points_per_voxel


# @numba.jit(nopython=True)
# def bound_points_jit(points, upper_bound, lower_bound):
#     # to use nopython=True, np.bool is not supported. so you need
#     # convert result to np.bool after this function.
#     N = points.shape[0]
#     ndim = points.shape[1]
#     keep_indices = np.zeros((N, ), dtype=np.int32)
#     success = 0
#     for i in range(N):
#         success = 1
#         for j in range(ndim):
#             if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
#                 success = 0
#                 break
#         keep_indices[i] = success
#     return keep_indices


@numba.jit(nopython=True)
def _points_to_pillar_kernel(points,
                            voxel_size,
                            coors_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=100,
                            max_voxels=5000):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    ndim = 2
    grid_size = (coors_range[2:] - coors_range[:2]) / voxel_size  # [-51.2, -51.2 51.2, 51.2] / [0.8,0.8] ->[128,128]
    grid_size = np.round(grid_size, 0,grid_size).astype(np.int32) #[128,128,1] 在numba中的写法不一样。

    coor = np.zeros(shape=(2, ), dtype=np.int32)
    voxel_num = 0
    for i in range(N):
        failed = False
        for j in range(ndim): # (X,Y) -> 128,128
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j]) # voxel_size=[0.8,0.8,8]
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue

        voxelidx = coor_to_voxelidx[coor[0], coor[1]] # coor_to_voxelidx 128,128 [-1,-1,-1, ..., -1]
        if voxelidx == -1: # new points in a pillar
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1 # voxel_num=1
            coor_to_voxelidx[coor[0], coor[1]] = voxelidx # -1 -> 0
            coors[voxelidx] = coor # coors[0]= coor
        num = num_points_per_voxel[voxelidx] # 0
        if num < max_points: # max_points:100
            voxels[voxelidx, num] = points[i] # [0,0] <- i
            num_points_per_voxel[voxelidx] += 1 #
    return voxel_num

def points_to_pillar(points, # input
                     voxel_size, # [0.8, 0.8]
                     coors_range, #[-51.2, -51.2, 51.2, 51.2]
                     max_points= 100,
                     max_voxels=5000): # total number of voxels: 128*128= 16384

    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)

    voxelmap_shape = (coors_range[2:] - coors_range[:2]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())

    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32) #6000*1
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32) #128*128*1 [-1,-1,-1...-1]
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype) # 5000*100*4
    coors = np.zeros(shape=(max_voxels, 2), dtype=np.int32) #5000*3

    voxel_num = _points_to_pillar_kernel(  # return 有点的柱子数
            points, voxel_size, coors_range, num_points_per_voxel,
            coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num] # voxel_num x3
    voxels = voxels[:voxel_num] #voxel_num x 100 x 4
    num_points_per_voxel = num_points_per_voxel[:voxel_num] #voxel_numx1

    return voxels, coors, num_points_per_voxel
