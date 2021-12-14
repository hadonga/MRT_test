import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import math

# pcDir = "D:/Projects/pointcloud/ourCodes/comparison/sequences/sequences/"
# LabPCDir = "E:/Datasets/SemanticKitti/dataset/Kitti/sequences"
# serverDir = "/root/dataset/kitti/sequences/" old server
serverDir = "/home/dong/dataset/sequences/" #new server

class kitti_loader(Dataset):
    def __init__(self, data_dir=serverDir,
                 train=True, skip_frames=0, npoints=100000):
        self.train = train
        self.data_dir = data_dir
        self.train = train
        self.skip_frames = skip_frames
        self.maxPoints = npoints

        self.proj_H = 64
        self.proj_W = 1024
        self.fov_up = 3
        self.fov_down = -25

        self.reset()
        self.load_filenames()

    def reset(self):
        self.pointcloud_path = []
        self.label_path = []

        self.complete_data=np.zeros((0, 4), dtype=np.float32)
        """ Reset scan members. """
        self.point = np.zeros((0, 3), dtype=np.float32)       # [m, 3]: x, y, z
        self.remission = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission
        self.label = np.zeros((0, 1), dtype=np.int32)     # [m, 1]: label

        self.proj_range = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32) # projected range image - [H,W] range (-1 is no data)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)  # unprojected range (list of depths for each point)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), 0, dtype=np.float32) # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32) # projected remission - [H,W] intensity (-1 is no data)

        # projected index (for each pixel, what I am in the pointcloud)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)  # [H,W] index (-1 is no data)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)  # [H,W] mask

        # projection color with semantic labels
        self.proj_sem_label = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)

        self.proj_cat=np.zeros((5,self.proj_H, self.proj_W), dtype=np.int32)

        #concatenate the mask
        self.masked_label=[]

        # sensor parameters
        self.img_means=[12.12,10.88,0.23,-1.04,0.21]
        self.img_stds=[12.32,11.47,6.91,0.86,0.16]

    def load_filenames(self):
        if self.train:
            seq = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
            # seq = ['08']
            for seq_num in seq:
                folder_pc = os.path.join(self.data_dir, seq_num, 'velodyne')
                folder_lb = os.path.join(self.data_dir, seq_num, 'labels')

                file_pc = os.listdir(folder_pc)
                file_pc.sort(key=lambda x: str(x[:-4]))
                file_lb = os.listdir(folder_lb)
                file_lb.sort(key=lambda x: str(x[:-4]))

                for index in range(0, len(file_pc), self.skip_frames+1):
                    self.pointcloud_path.append('%s/%s' % (folder_pc, file_pc[index]))
                    self.label_path.append('%s/%s' % (folder_lb, file_lb[index]))
        else:
            seq = '08'
            folder_pc = os.path.join(self.data_dir, seq, 'velodyne')
            folder_lb = os.path.join(self.data_dir, seq, 'labels')
            file_pc = os.listdir(folder_pc)
            file_pc.sort(key=lambda x: str(x[:-4]))
            file_lb = os.listdir(folder_lb)
            file_lb.sort(key=lambda x: str(x[:-4]))
            for index in range(0, len(file_pc), self.skip_frames+1):
                self.pointcloud_path.append('%s/%s' % (folder_pc, file_pc[index]))
                self.label_path.append('%s/%s' % (folder_lb, file_lb[index]))


    def get_data(self, pointcloud_path, label_path):
        # points = np.load(pointcloud_path) # for npy files
        data = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
        # print("File loaded : ", pointcloud_path)
        # self.point = data[:, 0:3]
        # self.remission = data[:, 3]
        label = np.fromfile(label_path, dtype=np.int32).reshape((-1,1))
        # print("Label loaded : ", label_path)

        sem_label = label & 0xFFFF  # semantic label in lower half

        concatData = np.hstack((data, sem_label))

        ###########################################################

        datalength = concatData.shape[0]

        if (datalength > self.maxPoints):
            concatData = concatData[:self.maxPoints, :]

        if (datalength < self.maxPoints):
            # print("difference", self.maxPoints - datalength)
            concatData=np.pad(concatData, [(0, self.maxPoints-datalength), (0, 0)], mode='constant')

        self.point = concatData[:,0:3]
        self.remission = concatData[:,3]
        self.label = concatData[:,4]
        # label = np.expand_dims(label, axis=0)
        # print("shape of label",np.shape(label))


        ###########################################################

        # rawdata = np.zeros((self.maxPoints, 5), dtype=np.int32)
        # datalength = concatData.shape[0]
        #
        # if (datalength <= self.maxPoints):
        #     rawdata[:datalength, :] = concatData
        # else:
        #     rawdata = concatData[:self.maxPoints, :]
        #
        # self.point = rawdata[:,0:3]
        # self.remission = rawdata[:,3]
        # self.label = rawdata[:,4]

        ###########################################################

        # if not isinstance(label, np.ndarray):
        #     raise TypeError("Label should be numpy array")
        # sem_label = label & 0xFFFF  # semantic label in lower half
        # inst_label = label >> 16  # instance id in upper half
        #
        # self.label = sem_label

    def limitDataset(self, xlim, ylim, zlim):
        # square
        # point = np.array([x for x in point if 0 < x[0] + 51.2 < 102.4 and 0 < x[1] + 51.2 < 102.4 and 0< x[2]+5 < 8])

        self.point = np.array([x for x in self.point if
                          0 < x[0] - xlim[0] < xlim[1] - xlim[0] and 0 < x[1] - ylim[0] < ylim[1] - ylim[0] and 0 < x[
                              2] - zlim[0] < zlim[1] - zlim[0]])

        # if len(self.point) >= self.maxPoints:
        #     choice = np.random.choice(len(self.point), self.maxPoints, replace=False)
        # else:
        #     choice = np.random.choice(len(self.point), self.maxPoints, replace=True)
        # self.point = self.point[choice]
        # self.label = self.label[choice]

    def limitDataset_circular(self, xylim, zlim):

        self.point = np.array(
            [x for x in self.point if math.sqrt(x[0] ** 2 + x[1] ** 2) < xylim and 0 < x[2] - zlim[0] < zlim[1] - zlim[0]])

        # if len(self.point) >= self.maxPoints:
        #     choice = np.random.choice(len(self.point), self.maxPoints, replace=False)
        # else:
        #     choice = np.random.choice(len(self.point), self.maxPoints, replace=True)
        #
        #
        # self.point = self.point[choice]
        # self.remission = self.remission[choice]
        # self.label = self.label[choice]


    def do_range_projection(self):
        # laser parameters
        fov_up = self.fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.point, 2, axis=1)

        # get scan components
        scan_x = self.point[:, 0]
        scan_y = self.point[:, 1]
        scan_z = self.point[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)
        pitch=np.nan_to_num(pitch)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.point[order]
        labels=self.label[order]

        # print("size of remission",len(self.remission))
        # print("Max val order",max(order))
        remission = self.remission[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)

        self.proj_sem_label[proj_y, proj_x] = labels



    def car_only_masking(self):
        lab = self.proj_sem_label * self.proj_mask
        proj_label_car = np.zeros((self.proj_H, self.proj_W), dtype=np.float32)
        proj_label_car[lab == 10] = 1
        proj_label_env = np.full((self.proj_H, self.proj_W), 1, dtype=np.float32)
        proj_label_env[lab == 10] = 0
        self.masked_label = np.stack((proj_label_car, proj_label_env), axis=0)
        self.masked_label = proj_label_car

    def ground_only_masking(self):
        lab = self.proj_sem_label * self.proj_mask
        proj_label_ground = np.zeros((self.proj_H, self.proj_W), dtype=np.float32)
        proj_label_ground[lab == 40] = 1
        proj_label_ground[lab == 44] = 1
        proj_label_ground[lab == 48] = 1
        proj_label_ground[lab == 49] = 1
        proj_label_ground[lab == 60] = 1
        # proj_label_ground[lab == 40 or lab == 44 or lab == 48 or lab == 49 or lab == 60] = 1
        self.masked_label = proj_label_ground

    def __len__(self):
        return len(self.pointcloud_path)



    def __getitem__(self, index):
        self.get_data(self.pointcloud_path[index], self.label_path[index])
        # self.limitDataset([-51.2, 51.2], [-51.2, 51.2], [-5, 3])
        # self.limitDataset_circular(39.9, [-3, 5])
        self.do_range_projection()

        rem = np.expand_dims(self.proj_remission, axis=0)
        range = np.expand_dims(self.proj_range, axis=0)
        xyz = np.rollaxis(self.proj_xyz, 2)
        projection_cat =np.concatenate((rem,range,xyz),axis=0)
        # rem = np.stack((self.proj_remission, self.proj_xyz), axis=2)
        # self.proj_xyz=np.reshape(self.proj_xyz,())
        # print("remission",rem.shape)
        # print("xyz", xyz.shape)
        # print("range", range.shape)
        self.proj_cat=projection_cat
        self.ground_only_masking()

        # self.car_only_masking()

        # print(self.point.shape)
        # unproj_n_points = self.point.shape[0]
        # p_x = np.full((1, self.maxPoints), -1, dtype=np.float32)
        # p_x[:unproj_n_points] = self.proj_x
        #
        # p_y = np.full((1, self.maxPoints), -1, dtype=np.float32)
        # p_y[:unproj_n_points] = self.proj_y

        # label = np.expand_dims(self.masked_label, axis=0)
        # lab = np.expand_dims(self.proj_sem_label, axis=0)
        # proj_mask = torch.from_numpy(self.proj_mask)
        # self.proj_sem_label = torch.from_numpy(self.proj_sem_label).clone()
        # lab = self.proj_sem_label * proj_mask
        return projection_cat, self.masked_label, self.unproj_range,self.proj_x,self.proj_y,self.point,self.remission,self.label,self.proj_mask
        # return self.proj_remission,self.proj_sem_label,self.proj_xyz, self.proj_range, self.proj_idx, self.proj_mask