import numpy as np
import math

#点转扇形网格

def points_to_cylinder(points, #
                     voxel_size=[3.6,0.5,8], # [3.6, 0.5, 8] 表示扇形旋角度为3.6度，扇形水平距离为1米，高度为8米
                     dist=50, #一个值，点距离激光雷达的最大水平距离
                     max_points=35, #扇形网格中点的最大数量
                     max_cylinder=10000):
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype) #3.6,0.5,8

    coors_range=np.array([360, dist , 8])
    voxelmap_shape = coors_range / voxel_size # 100x100x1
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist()) # 100x100x1

    num_points_per_cylinder = np.zeros(shape=(max_cylinder, ), dtype=np.int32) # 10000

    coor_to_cylinderidx = -np.ones(shape=voxelmap_shape, dtype=np.int32) # 100x100

    cylinder = np.zeros(
        shape=(max_cylinder, max_points, points.shape[-1]), dtype=points.dtype) #10000*35*4
    coors = np.zeros(shape=(max_cylinder, 3), dtype=np.int32) #10000*3
    cylinder_num = 0
    N = points.shape[0] #Number of points

    #
    for i in range(N):
        coor = np.zeros(shape=(3,), dtype=np.int32)
        # segid=np.floor(math.atan2(points[i,1],points[i,0]) + math.pi/(voxel_size[0]/180*math.pi)) #????

        ang_id=np.floor((math.atan2(points[i,1],points[i,0]) /math.pi*(180/voxel_size[0]))+50)  # 起始点从第三象限开始逆时针旋转(0->99)
        rad_id=np.floor(math.sqrt(points[i,0]**2+points[i,1]**2)/voxel_size[1]) #
        height_id=np.floor((points[i,2]+voxel_size[2]/2)/voxel_size[2])

        if ang_id >= 100:
            print(str(i)+ "ang_id out of range")
        elif rad_id >= 100:
            print(str(i)+"rad_id out of range")


        coor[0]=ang_id
        coor[1]=rad_id
        coor[2]=height_id

        cylinderidx = coor_to_cylinderidx[coor[0], coor[1], coor[2]]

        if cylinderidx==-1:
            cylinderidx = cylinder_num
            if cylinder_num >= max_cylinder:
                break
            cylinder_num+=1
            coor_to_cylinderidx[coor[0], coor[1], coor[2]] = cylinderidx #占位
            coors[cylinderidx] = coor
        num = num_points_per_cylinder[cylinderidx]

        if num < max_points:
            cylinder[cylinderidx, num] = points[i]
            num_points_per_cylinder[cylinderidx] += 1

    pts_in_cylinder = cylinder[:cylinder_num]  # p x n x 4
    cylinder_coors=coors[:cylinder_num] # p x 3
    num_points_per_cylinder = num_points_per_cylinder[:cylinder_num] # p x 1
    return pts_in_cylinder, cylinder_coors, num_points_per_cylinder  # 共同点是 p