import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


pointcloud_path = 'E:/Datasets/NuScenes/v1.0-mini'

filename="n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"
file2 ="n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007396978.pcd.bin"

load_point_cloud = "E:/Datasets/NuScenes/v1.0-mini/sweeps/LIDAR_TOP/"+file2

pc1=np.fromfile(load_point_cloud,dtype=int)

nusc = NuScenes(version='v1.0-mini', dataroot=pointcloud_path, verbose=True)
# print("Total number of scenes:", len(nusc.scene))






split_dir='E:\TestCodes\MotionNet\MotionNet\data\split.npy'

scenes = np.load(split_dir, allow_pickle=True).item().get('train')
print("Split: {}, which contains {} scenes.".format('train', len(scenes)))

res_scenes = list()
for s in scenes:
    s_id = s.split('_')[1]
    res_scenes.append(int(s_id))


scene_idx=0

num_keyframe_skipped = 0  # The number of keyframes we will skip when dumping the data
nsweeps_back = 30  # Number of frames back to the history (including the current timestamp)
nsweeps_forward = 20  # Number of frames into the future (does not include the current timestamp)
skip_frame = 0  # The number of frames skipped for the adjacent sequence
num_adj_seqs = 2  # number of adjacent sequences, among which the time gap is \delta t


curr_scene = nusc.scene[scene_idx]


first_sample_token = curr_scene['first_sample_token']
curr_sample = nusc.get('sample', first_sample_token)
curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])

save_data_dict_list = list()  # for storing consecutive sequences; the data consists of timestamps, points, etc
save_box_dict_list = list()  # for storing box annotations in consecutive sequences
save_instance_token_list = list()
adj_seq_cnt = 0
save_seq_cnt = 0  # only used for save data file name


# Iterate each sample data
print("Processing scene {} ...".format(scene_idx))





# while curr_sample_data['next'] != '':

# Get the synchronized point clouds
all_pc, all_times, trans_matrices = \
    LidarPointCloud.from_file_multisweep_bf_sample_data(nusc, curr_sample_data,
                                                        return_trans_matrix=True,
                                                        nsweeps_back=nsweeps_back,
                                                        nsweeps_forward=nsweeps_forward)

pc = all_pc.points


print("test")