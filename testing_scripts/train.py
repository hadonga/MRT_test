from load_data import data_loader
import yaml
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# loading arch and kitti files
archfile = 'config/arch/initial_arch.yaml'
configfile = 'config/labels/semantickitti/semantic-kitti-color-mod.yaml'

with open(archfile, 'r') as stream:
    param = yaml.safe_load(stream)

with open(configfile, 'r') as stream:
    kitti_config = yaml.safe_load(stream)

proj_W=1024
proj_H=64

def show_2D_range( range_2D,title, scale,show_plot=True, save_image=False, path=''):

    color_map = kitti_config['color_map']
    plt.figure(figsize=((proj_W/scale)-2, (proj_H/scale)+0.5))
    plt.title(title)
    image = np.array([[color_map[val] for val in row] for row in range_2D], dtype='B')
    plt.imshow(image)



def show_2D_image( image,title,scale, colormap="magma"):
    cmap = plt.cm.get_cmap(colormap, 10)
    plt.figure(figsize=((proj_W/scale)-2, (proj_H/scale)+0.5))
    plt.title(title)
    plt.imshow(image, cmap=cmap,)
    plt.clim(0, 80)
    plt.colorbar()
    plt.show()



# dataset = data_loader(param, kitti_config, train='train')

train_dataset = data_loader(param, kitti_config, train='train')

# range = train_dataset[75]

data, labels, unproj_range, p_x, p_y, curr_points, projected_curr_range, curr_lab,proj_inst,curr_inst = train_dataset[75]


# saving
# np.save("data.npy",data)
# np.save("labels.npy",labels)
# np.save("unproj_range.npy",unproj_range)
# np.save("p_x.npy",p_x)
# np.save("p_y.npy",p_y)
# np.save("curr_points.npy",curr_points)
# np.save("projected_curr_range",projected_curr_range)
# np.save("curr_lab.npy",curr_lab)
# np.save("proj_inst.npy",proj_inst)
# np.save("curr_inst.npy",curr_inst)


# for u in range(3):
#     for v in range(5):
#         show_2D_image(data[u][v][1],str(u)+"-"+str(v),40)
#
#
# show_2D_image(projected_curr_range,"original_range",40)
# show_2D_range(labels,"Labels",40)



# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False,
#                             num_workers=0, pin_memory=True, drop_last=True)

# for batch_idx, (data,labels,unproj_range,p_x,p_y,curr_points,curr_rem,curr_lab) in enumerate(train_dataloader):
#     print("batch : ",batch_idx)


# concat_time_frames,label_onehot,unproj_range,p_x,p_y,curr_points,curr_rem,curr_lab = dataset[5599]


print("Test")