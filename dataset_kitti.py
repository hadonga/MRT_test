import os
from os.path import join
import numpy as np
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import math
import copy
import numba


# A class to load the kitti dataset file names

class kitti_loader():
    def __init__(self, data_dir, kitti_param, kitti_config, step=''):

        self.dir = data_dir
        self.train = step  # 'train','validate','test'
        self.skip_frames = kitti_param['skip_frames']
        self.calibrate = kitti_param['calibrate']

        self.pointcloud_path = []
        self.label_path = []

        self.train_seq = kitti_config['split']['train']
        self.valid_seq = kitti_config['split']['valid']
        self.test_seq = kitti_config['split']['test']

        self.calibrations = []
        self.times = []
        self.poses = []

        self.load_filenames()

    def load_a_seq(self, seq_num, i):
        folder_pc = join(self.dir, seq_num, 'velodyne')
        folder_lb = join(self.dir, seq_num, 'labels')

        file_pc = os.listdir(folder_pc)
        file_pc.sort(key=lambda x: str(x[:-4]))
        if self.train != 'test':
            file_lb = os.listdir(folder_lb)
            file_lb.sort(key=lambda x: str(x[:-4]))

        if self.calibrate:
            self.calibrations.append(self.parse_calibration(join(self.dir, seq_num, "calib.txt")))  # read caliberation
            self.times.append(np.loadtxt(join(self.dir, seq_num, 'times.txt'), dtype=np.float32))  # read times
            poses_f64 = self.parse_poses(join(self.dir, seq_num, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])  # read poses

        for index in range(0, len(file_pc), self.skip_frames + 1):
            self.pointcloud_path[i].append('%s/%s' % (folder_pc, file_pc[index]))
            if self.train != 'test':
                self.label_path[i].append('%s/%s' % (folder_lb, file_lb[index]))

    def load_filenames(self):
        if self.train == 'train':
            sequences = self.train_seq
        elif self.train == 'validate':
            sequences = self.valid_seq
        elif self.train == 'test':
            sequences = self.test_seq
        else:
            raise ValueError("Please specify  : 'train' 'validate' or 'test'")

        for index, name in enumerate(sequences):
            self.pointcloud_path.append([])
            self.label_path.append([])
            self.load_a_seq(name, index)

    def parse_calibration(self, filename):
        calib = {}
        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            calib[key] = pose
        calib_file.close()
        return calib

    def parse_poses(self, filename, calibration):
        file = open(filename)
        poses = []
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)
        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        return poses

    def get_filenames(self):
        return self.pointcloud_path, self.label_path

    def get_poses(self):
        return self.poses

    def get_timestamps(self):
        return self.times


# A class to load data from files and pre-process

class Dataset_kitti(Dataset):
    def __init__(self, data_dir, param, kitti_config, step=''):

        self.train = step
        self.train_seq = kitti_config['split']['train']
        self.valid_seq = kitti_config['split']['valid']
        self.test_seq = kitti_config['split']['test']

        self.maxPoints = param['max_points']
        self.timeframes = param['timeframes']
        self.intervals = param['intervals']
        self.channels = param['input_channels']  # need to change if we add more features

        self.proj_H = param['sensor']['img_prop']['height']
        self.proj_W = param['sensor']['img_prop']['width']
        self.fov_up = param['sensor']['fov_up']
        self.fov_down = param['sensor']['fov_down']

        self.learning_map = kitti_config['learning_map']
        self.color_map = kitti_config['color_map']
        self.classes = len(kitti_config['learning_map_inv'])
        kittiloader = kitti_loader(data_dir, param, kitti_config, step)
        self.pointcloud_path, self.label_path = kittiloader.get_filenames()
        self.poses = kittiloader.get_poses()
        # timestamps = kitti.get_timestamps()
        self.fr_within_a_seq()

        # sensor parameters
        self.img_means = param['sensor']['img_means']
        self.img_stds = param['sensor']['img_stds']

        self.seq = 0
        self.frame = 0

        self.reset()

    def reset(self):

        self.complete_data = np.zeros((0, 4), dtype=np.float32)
        """ Reset scan members. """
        self.point = np.zeros((0, 3), dtype=np.float32)
        self.remission = np.zeros((0, 1), dtype=np.float32)
        self.label = np.zeros((0, 1), dtype=np.int32)
        self.label_ins = np.zeros((0, 1), dtype=np.int32)

        self.label_ord = np.zeros((0, 1), dtype=np.int32)
        self.depth = np.zeros((0, 1), dtype=np.float32)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)  # projected range image - [H,W] range (-1 is no data)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)  # unprojected range (list of depths for each point)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)  # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)  # projected remission - [H,W] intensity (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)  # [H,W] index (-1 is no data)
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y
        # self.proj_y_ord = np.zeros((0, 1), dtype=np.int32)
        # self.proj_x_ord = np.zeros((0, 1), dtype=np.int32)
        self.proj_mask = np.zeros((self.proj_H, self.proj_W), dtype=np.int32)  # [H,W] mask
        self.proj_sem_label = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        self.proj_ins_label = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        self.proj_cat = np.zeros((5, self.proj_H, self.proj_W), dtype=np.int32)
        self.masked_label = []

    # divide the index ->  into frame and seq since we are dealing temporal data so
    # working only with index will cause trouble in transition between sequences

    def fr_within_a_seq(self):
        frames_in_a_seq = []
        for i in range(len(self.pointcloud_path)):
            frames_in_a_seq.append(len(self.pointcloud_path[i]))
        self.frames_in_a_seq = np.array(frames_in_a_seq).cumsum()

    ########################################################################################################
    #                                 Data acquisition                                                     #
    ########################################################################################################

    def get_data(self, pointcloud_path, label_path):

        data = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
        if self.train != 'test':
            label = np.fromfile(label_path, dtype=np.int32).reshape((-1, 1))
            label = label & 0xFFFF
            label_ins = label >> 16

        else:
            label = np.zeros_like(data[:, 1], dtype=int)
            label = np.expand_dims(label, 1)
            label_ins = np.zeros_like(data[:, 1], dtype=int)
            label_ins = np.expand_dims(label_ins, 1)

        concatData = np.hstack((data, label, label_ins))

        datalength = concatData.shape[0]

        if (datalength > self.maxPoints):
            concatData = concatData[:self.maxPoints, :]

        if (datalength < self.maxPoints):
            concatData = np.pad(concatData, [(0, self.maxPoints - datalength), (0, 0)], mode='constant')

        # return data on order of xyz,intensity,semantic labels, instance labels each of length 100,000

        return concatData[:, 0:3], concatData[:, 3], concatData[:, 4].astype(int), concatData[:, 5].astype(int)

    def set_data(self, point, remission, label, label_ins):
        self.point, self.remission, self.label, self.label_ins = point, remission, label, label_ins

    # if we want to limit the dataset in a cubicical area
    def limitDataset_rectangular(self, xlim, ylim, zlim):
        self.point = np.array([x for x in self.point if
                               0 < x[0] - xlim[0] < xlim[1] - xlim[0] and 0 < x[1] - ylim[0] < ylim[1] - ylim[0] and 0 <
                               x[
                                   2] - zlim[0] < zlim[1] - zlim[0]])

    # if we want to limit the dataset in a cylinderical area
    def limitDataset_cylindrical(self, xylim, zlim):
        self.point = np.array(
            [x for x in self.point if
             math.sqrt(x[0] ** 2 + x[1] ** 2) < xylim and 0 < x[2] - zlim[0] < zlim[1] - zlim[0]])

    ########################################################################################################
    #                                 Range Projection                                                     #
    ########################################################################################################

    # function to convert 3D points + labels into -> range image [complete]
    def spherical_projection(self):

        fov_up = self.fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
        depth = np.linalg.norm(self.point, 2, axis=1)

        scan_x = self.point[:, 0]
        scan_y = self.point[:, 1]
        scan_z = self.point[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)
        pitch = np.nan_to_num(pitch)

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
        self.proj_y = np.copy(proj_y)  # store a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]

        # copy of depth in descending order
        self.depth = np.copy(depth)

        indices = indices[order]
        points = self.point[order]
        self.label_ord = self.label[order]
        self.label_ins = self.label_ins[order]

        remission = self.remission[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # self.proj_y_ord= np.copy(proj_y)
        # self.proj_x_ord = np.copy(proj_x)

        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)
        self.proj_sem_label[proj_y, proj_x] = self.label_ord
        self.proj_ins_label[proj_y, proj_x] = self.label_ins

    # function to concatenate projected features in shape 5*H*W [complete]
    def get_range_image(self):
        rem = np.expand_dims(self.proj_remission, axis=0)
        proj_range = np.expand_dims(self.proj_range, axis=0)
        xyz = np.rollaxis(self.proj_xyz, 2)
        projection_cat = np.concatenate((rem, proj_range, xyz), axis=0)

        return projection_cat

    ########################################################################################################
    #                                 Multi range Projection                                               #
    ########################################################################################################

    # Function to apply muli-range mask depending upon  lower and uper limit
    # Note that : for multi image projection, first the spherical_projection() function must have called
    # This function just apply range mask within the limits

    def apply_range_limits(self, lower, upper, get_label=True):

        rem = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)
        proj_sem_label = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.int32)

        mask = np.logical_and(self.proj_range > lower, self.proj_range < upper)
        minus_one_mask = self.proj_remission == -1

        rem = np.multiply(self.proj_remission, mask)
        proj_range = np.multiply(self.proj_range, mask)
        proj_xyz = np.multiply(np.rollaxis(self.proj_xyz, 2), mask)
        #
        # rem_new= rem -(1*minus_one_mask)
        # proj_range_new= proj_range- (1*minus_one_mask)
        # proj_xyz_new = proj_xyz-(1*minus_one_mask)

        rem = np.multiply(self.proj_remission, mask) - (1 * minus_one_mask)
        proj_range = np.multiply(self.proj_range, mask) - (1 * minus_one_mask)
        proj_xyz = np.multiply(np.rollaxis(self.proj_xyz, 2), mask) - (1 * minus_one_mask)

        # self.show_2D_image(range, "frame " + str(lower) + " - " + str(upper), 40, colormap="magma")

        if get_label:
            proj_sem_label = self.proj_sem_label
            proj_sem_label = np.multiply(proj_sem_label, mask)
            proj_sem_label[minus_one_mask]=self.classes
            # proj_sem_label = np.multiply(proj_sem_label, mask) - (1 * minus_one_mask)
            # self.show_2D_range(proj_sem_label, "range " + str(lower) + " - " + str(upper), 40, True, False, '')
            # proj_sem_label[self.proj_sem_label==-1]=-1
            # self.show_2D_range(proj_sem_label, "range " + str(lower) + " - " + str(upper), 40, True, False, '')
            # proj_sem_label = np.where(proj_sem_label==0,-1,proj_sem_label)
            # self.show_2D_range(self.proj_sem_label, "range " + str(lower) + " - " + str(upper), 40, True, False, '')
            # self.show_2D_range(proj_sem_label, "range " + str(lower) + " - " + str(upper), 40, True, False, '')

        projection_cat = np.concatenate((np.expand_dims(rem, axis=0), np.expand_dims(proj_range, axis=0), proj_xyz),
                                        axis=0)

        return projection_cat, proj_sem_label

    # Testing function for single range projection

    def single_range_projection(self):
        self.spherical_projection()
        min_depth = 2
        max_depth = 60

        #
        # # Divide the depth into multiple ranges
        # concated_proj_range = np.zeros((0, self.channels, self.proj_H, self.proj_W), dtype=np.float32)
        # concated_proj_label = np.zeros((0, self.proj_H, self.proj_W), dtype=np.int32)

        if self.train != 'test':
            lab = True
        range_output, label_output = self.apply_range_limits(min_depth, max_depth,
                                                             lab)  # range_output = C*H*W ; label_output = 1*H*W
        return range_output, label_output

    # This function is to apply range limits and get the concatenated outputs of multi-range

    def multi_range_projection(self):
        self.spherical_projection()

        max_iter, min_depth, max_depth = len(self.intervals) + 1, min(self.depth), max(self.depth)

        # Divide the depth into multiple ranges
        concated_proj_range = np.zeros((0, self.channels, self.proj_H, self.proj_W), dtype=np.float32)
        concated_proj_label = np.zeros((0, self.proj_H, self.proj_W), dtype=np.int32)

        if self.train != 'test':
            lab = True

        for index in range(max_iter):
            if index == 0:
                range_output, label_output = self.apply_range_limits(min_depth, self.intervals[index],
                                                                     lab)  # range_output = C*H*W ; label_output = 1*H*W
            elif index == max_iter - 1:
                range_output, label_output = self.apply_range_limits(self.intervals[index - 1], max_depth, lab)
            else:
                range_output, label_output = self.apply_range_limits(self.intervals[index - 1], self.intervals[index],
                                                                     lab)
            concated_proj_range = np.concatenate((concated_proj_range, np.expand_dims(range_output, 0)), axis=0)
            concated_proj_label = np.concatenate((concated_proj_label, np.expand_dims(label_output, 0)), axis=0)

        return concated_proj_range, concated_proj_label

    ########################################################################################################
    #                                 Other Functions                                                      #
    ########################################################################################################

    # This function is to map higher number of classes to lower classes

    def class_mapping(self, semlabels):
        original_map = self.learning_map
        learning_map = np.zeros((np.max([k for k in original_map.keys()]) + 1), dtype=np.int32)
        for k, v in original_map.items():
            learning_map[k] = v

        return learning_map[semlabels]

    # get sequence and frame -> from index
    def get_seq_and_frame(self, index):

        if index < self.frames_in_a_seq[0]:
            return 0, index
        else:
            seq_count = len(self.frames_in_a_seq)
            for i in range(seq_count):
                fr = index + 1
                if i < seq_count - 1 and self.frames_in_a_seq[i] < fr and self.frames_in_a_seq[i + 1] > fr:
                    # print("here")
                    return i + 1, index - self.frames_in_a_seq[i]

                elif i < seq_count - 1 and self.frames_in_a_seq[i] == fr:
                    return i, index - self.frames_in_a_seq[i - 1]

                elif i < seq_count - 1 and fr == self.frames_in_a_seq[-1]:
                    return seq_count - 1, index - self.frames_in_a_seq[-2]

    # set the sequence and frame

    def set_seq_and_frame(self, index):

        self.seq, self.frame = self.get_seq_and_frame(index)

        if self.train == 'train':
            seq_name = self.train_seq[self.seq]
        elif self.train == 'validate':
            seq_name = self.valid_seq[self.seq]
        elif self.train == 'test':
            seq_name = self.test_seq[self.seq]
        else:
            raise ValueError("Please enter correct one e.g. 'train','validate' or 'test' ")

        # print("seq is : ", seq_name, " and frame is : ", self.frame, "path is : ",self.pointcloud_path[self.seq][self.frame])

    # 2D visualization of labels -> discrete values->colors provided by semantic kitti

    def show_2D_range(self, range_2D, title, scale, show_plot=True, save_image=False, path=''):

        plt.figure(figsize=((self.proj_W / scale) - 2, (self.proj_H / scale) + 0.5))
        plt.title(title)
        image = np.array([[self.color_map[val] for val in row] for row in range_2D], dtype='B')
        plt.imshow(image)

    # 2D visualization any image

    def show_2D_image(self, image, title, scale, colormap="magma"):
        cmap = plt.cm.get_cmap(colormap, 10)
        plt.figure(figsize=((self.proj_W / scale) - 2, (self.proj_H / scale) + 0.5))
        plt.title(title)
        plt.imshow(image, cmap=cmap)
        plt.colorbar()
        plt.show()

    # seq, frame -> path

    def get_curr_paths(self):
        if self.train != 'test':
            return self.pointcloud_path[self.seq][self.frame], self.label_path[self.seq][self.frame]
        else:
            return self.pointcloud_path[self.seq][self.frame], ''

    def get_paths(self, seq, frame):
        if self.train != 'test':
            return self.pointcloud_path[seq][frame], self.label_path[seq][frame]
        else:
            return self.pointcloud_path[seq][frame], ''

    def all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    # get the length of dataset i.e. number of frames[in all sequences]

    def __len__(self):
        return self.frames_in_a_seq[-1]

    # Get data

    def __getitem__(self, index):

        if index > len(self):
            raise ValueError("Please enter the index within range")

        # self.set_seq_and_frame(index)
        # points_path, label_path = self.get_paths(self.seq, self.frame)
        # points, remission, label, label_ins = self.get_data(points_path, label_path)
        # label = self.class_mapping(label)
        # self.set_data(points, remission, label, label_ins)
        # self.spherical_projection()
        # proj_label, proj_range, unproj_range, p_x, p_y, pts, rem, lab = self.get_range_param()
        #
        # return proj_range

        # shape -> time frames , no of multi ranegs, features [channels], height , width
        concat_time_frames = np.zeros((0, len(self.intervals) + 1, self.channels, self.proj_H, self.proj_W),
                                      dtype=np.float32)
        concat_labels = np.zeros((0, len(self.intervals) + 1, self.proj_H, self.proj_W), dtype=np.int32)

        # return concat_time_frames,self.ori_proj_label,self.ori_proj_range,self.ori_unproj_range,self.ori_p_x,self.ori_p_y,self.ori_pts,self.ori_rem,self.ori_lab

        if self.timeframes > 1:
            self.set_seq_and_frame(index)
            pose0 = self.poses[self.seq][self.frame]  # reference pose

            for ite in range(self.timeframes):
                # Loop will run for t times where t is number of time frames

                reverse_ite = self.timeframes - (ite + 1)  # reverse index (only for testing)

                if self.frame - ite >= 0:
                    curr_frame = self.frame - ite
                else:
                    print("getting frame 0 as reference")
                    curr_frame = 0

                pose = self.poses[self.seq][curr_frame]  # current pose

                points_path, label_path = self.get_paths(self.seq, curr_frame)

                # Load points of each frame and apply axis transformation according to reference frame

                points, remission, label, label_ins = self.get_data(points_path, label_path)
                hpoints = np.hstack((points, np.ones_like(points[:, :1])))
                new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
                new_points = new_points[:, :3]

                if ite == 0:
                    new_coords = points[:, :]

                else:
                    new_coords = new_points - pose0[:3, 3]
                    new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
                    new_coords = np.hstack((new_coords, points[:, 3:]))

                label = self.class_mapping(label)  # lowering the number of classes in labels

                self.set_data(new_coords, remission, label, label_ins)

                concated_proj_range, concated_proj_label = self.multi_range_projection()  # get the multi range data

                single_proj_range, single_proj_label = self.single_range_projection()

                # concated_proj_range, _ = self.multi_range_projection()

                if ite == 0:
                    # saving the data for frame at time = t

                    # other output save
                    # unproj_n_points = points.shape[0]
                    # p_x = np.full((1, unproj_n_points), -1, dtype=np.int32)
                    # p_x[:unproj_n_points] = self.proj_x
                    # p_y = np.full((1, unproj_n_points), -1, dtype=np.int32)
                    # p_y[:unproj_n_points] = self.proj_y
                    p_x, p_y = copy.deepcopy(self.proj_x), copy.deepcopy(self.proj_y)

                    unproj_range, curr_points, curr_rem, curr_lab = copy.deepcopy(self.unproj_range), \
                                                                    copy.deepcopy(points), copy.deepcopy(
                        remission), copy.deepcopy(label)

                    proj_range_total = copy.deepcopy(self.proj_range)
                    proj_single_label = copy.deepcopy(self.proj_sem_label)
                    proj_multi_label = copy.deepcopy(concated_proj_label)
                    # print("within function" , self.proj_range[11][0:40])
                    # label_onehot = np.zeros(self.proj_sem_label.shape + (self.classes,), dtype=int)
                    # label_onehot[self.all_idx(self.proj_sem_label, axis=2)] = 1
                    # label_onehot=np.rollaxis(label_onehot,2)

                concat_time_frames = np.concatenate((concat_time_frames,
                                                     np.expand_dims(concated_proj_range, 0)), axis=0)

                # concat_labels = np.concatenate((concat_labels, np.expand_dims(concated_proj_label,0)), axis=0)

            # return
            # concat_time_frames,
            # projected_label,
            # unproj_range,
            # p_x, p_y,
            # curr_points,
            # projected_curr_range,
            # curr_lab
            return {"data": concat_time_frames,  # pixel features - size: BxTxRx5xHxW
                    "gt_multi_pixel": proj_multi_label,# single pixel label - size: Bx5xHxW
                    "single_data": concat_time_frames[0][0].astype(np.float32),  # pixel features -size: Bx5xHxW
                    "gt_single_pixel": proj_multi_label[0].astype(np.float32),  # single pixel label - size: BxHxW

                    # "unproj_range": unproj_range, # range information of points - size: Bx100,000x1
                    # "p_x": p_x,"p_y": p_y, # point location in image - size: Bx100,000
                    # "points":curr_points, # point features - size : size: Bx100,000x3
                    # "proj_range":proj_range_total, # point->pixel - size: BxHxW
                    # "groundtruth_points": curr_lab # original gt of points - size : Bx100,000x1
                    }

        else:
            raise ValueError("Please enter the number of time frames > 1")
