import numpy as np
# from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import yaml


def show_2D_image( image,W,H, cmap="tab20c"):
    # cmap = plt.cm.get_cmap(cmap, 10)
    plt.figure(figsize=((W/40) - 2, (H/40) + 0.5))
    plt.imshow(image)
    # plt.clim(0, 80)
    plt.colorbar()
    plt.show()



def show_2D_range(range_2D,W,H,color_map):
    plt.figure(figsize=((W/40) - 2, (H/40) + 0.5))
    # plt.title(title)
    image = np.array([[color_map[val] for val in row] for row in range_2D], dtype='B')
    plt.imshow(image)
    plt.show()


def get_data(pointcloud_path, label_path):

    data = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
    label_org = np.fromfile(label_path, dtype=np.int32).reshape((-1,1))
    label = label_org & 0xFFFF
    label_ins = label_org >> 16

    concatData = np.hstack((data, label,label_ins))
    datalength = concatData.shape[0]

    if (datalength > 100000):
        concatData = concatData[:100000, :]

    if (datalength  < 100000):
        concatData = np.pad(concatData, [(0, 100000 - datalength), (0, 0)], mode='constant')

    return concatData[:,0:3],concatData[:,3],concatData[:,4].astype(int),concatData[:,5].astype(int)


def class_mapping(learning_map,semlabels):
    original_map = learning_map
    learning_map = np.zeros((np.max([k for k in original_map.keys()]) + 1), dtype=np.int32)
    for k, v in original_map.items():
        learning_map[k] = v

    return learning_map[semlabels]



# data_path='E:/Datasets/SemanticKitti/dataset/Kitti/sequences/00/velodyne/000075.bin'
# label_path='E:/Datasets/SemanticKitti/dataset/Kitti/sequences/00/labels/000075.label'
#
# point,_,label,inst_label=get_data(data_path,label_path)
#
#
# configfile = 'config/labels/semantickitti/semantic-kitti-color-mod.yaml'
# with open(configfile, 'r') as stream:
#     kitti_config = yaml.safe_load(stream)
#
#
# learning_map=kitti_config['learning_map']
# label=class_mapping(learning_map,label)


xyz=np.load("points.npy")
# tree = KDTree(xyz, leaf_size=2)
labels =np.load("labels.npy")
proj_labels=np.load("proj_labels.npy")
point_inst = np.load("curr_inst.npy")
proj_inst = np.load("proj_inst.npy")


# car_only_inst=labels
car_only_inst = np.zeros((np.shape(point_inst)))
mask=labels==1
car_only_inst=point_inst*mask
car_only_label=labels*mask



x=xyz[:,0]
y=xyz[:,1]
z=xyz[:,2]

xlimit=[-80,80]
ylimit=[-16,32]
zlimit=[-3.5,3]

x=x-xlimit[0]
y=y-ylimit[0]

x = x/(xlimit[1]-xlimit[0])
y = y/(ylimit[1]-ylimit[0])

f=10
W=(xlimit[1]+abs(xlimit[0]))*f
H=(ylimit[1]+abs(ylimit[0]))*f

x = np.floor(x*W).astype(np.int32)
y = np.floor(y*H).astype(np.int32)

proj_x = np.minimum(W - 1, x)
proj_x = np.maximum(0, proj_x)

proj_y = np.minimum(W - 1, y)
proj_y = np.maximum(0, proj_y)




BEV_labels = np.full((H,W),-1, dtype=np.int32)
BEV_labels[proj_y,proj_x]=labels

BEV_labels_car = np.full((H,W),-1, dtype=np.int32)
BEV_labels_car[proj_y,proj_x]=car_only_label

# un,v=np.unique(BEV_labels_car,True)

BEV_z = np.full((H,W),zlimit[0], dtype=np.float64)
BEV_z[proj_y,proj_x]=z

BEV_ins_labels = np.full((H,W),-1, dtype=np.int32)
BEV_ins_labels[proj_y,proj_x]=car_only_inst

unique=np.unique(BEV_ins_labels)

configfile = 'config/labels/semantickitti/colormap.yaml'
with open(configfile, 'r') as stream:
    kitti_config = yaml.safe_load(stream)
color_map=kitti_config['color_map']

custom_colorbar ={}

# dict.fromkeys(unique.astype(str), [0,0,0])

for ind in range(len(unique)):
    color = list(np.random.choice(range(256), size=3))
    custom_colorbar[unique[ind]]=color


custom_colorbar[-1]=[0,0,0]
custom_colorbar[0]=[255,255,255]

show_2D_range(BEV_labels,W,H,color_map)
# show_2D_range(BEV_labels_car,W,H,color_map)
show_2D_range(BEV_ins_labels,W,H,custom_colorbar)
# show_2D_range(BEV_ins_labels,W,H,custom_colorbar)

# show_2D_image(BEV_ins_labels,W,H)
# show_2D_image(BEV_z,W,H)

# show_2D_range(proj_labels,1024,64,color_map)


labels=np.load("labels.npy")
px=np.load("px.npy")
py=np.load("py.npy")
W=1024
H=64

idx_list = py * W + px
newLabels =labels.reshape((W*H),1)
newLabels1 = newLabels[idx_list]


print("test")