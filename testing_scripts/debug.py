import numpy as np
import matplotlib.pyplot as plt


# def show_2D_range(range_2D, title, scale, show_plot=True, save_image=False, path=''):
#     plt.figure(figsize=((1024 / scale) - 2, (64 / scale) + 0.5))
#     plt.title(title)
#     image = np.array([[color_map[val] for val in row] for row in range_2D], dtype='B')
#     plt.imshow(image)
#     if save_image:
#         plt.savefig(path)
#     if show_plot:
#         plt.show()


def show_2D_image(image, title, scale,i,j,colormap="magma"):
    cmap = plt.cm.get_cmap(colormap, 10)
    plt.figure(figsize=((1024 / scale) - 2, (64 / scale) + 0.5))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.colorbar()
    plt.savefig(str(i)+"_"+str(j)+".png")
    plt.show()


concat_time_frames=np.load("concat_time_frames.npy")
concat_labels=np.load("concat_labels.npy")

for i in range (3):
    for j in range (5):
        show_2D_image(concat_time_frames[i][j][1],str(i)+","+str(j),40,i,j)

print("test")


