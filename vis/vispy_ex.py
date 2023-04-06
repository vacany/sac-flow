import os

import numpy as np
import vispy.scene
from vispy.scene.visuals import Arrow
from vispy.scene import visuals
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# https://vispy.org/gallery/scene/complex_image.html

# this will work
# Make a canvas and add simple view
#
os.makedirs(os.path.expanduser('~') + '/tmp', exist_ok=True)
# def visualize_points3D(points, labels=None, point_size=0.01, **kwargs):


import sys
#TODO make interface and swapping the frames, baseline?
frame = 162  # 19 ICP failed
data = np.load(f'/home/patrik/rci/pretrained_models/kitti_v_100_examples/pc_res/{frame:06d}_res..npz', allow_pickle=True)
# data = np.load(f'/home/patrik/rci/experiments/scoop_vis_smooth/pc_res/{frame:06d}_res.npz', allow_pickle=True)

pc1 = data['pc1'][:, [0, 2, 1]]
pc2 = data['pc2'][:, [0, 2, 1]]
gt_mask1 = data['gt_mask_for_pc1']
gt_flow1 = data['gt_flow_for_pc1'][:, [0, 2, 1]]
est_flow1 = data['est_flow_for_pc1'][:, [0, 2, 1]]
corr_conf1 = data['corr_conf_for_pc1']

# for flow
points = pc1
vis_flow = gt_flow1

# visual scenes result scenes. Find worst cases recursively?
# order input list by shit epe

# for multiple?
points = np.concatenate((pc1, pc2), axis=0)
vis_flow = np.concatenate((gt_flow1, np.zeros(pc2.shape)), axis=0)
all_rays = []

for x in range(1, 20):
    ray_points = points + ((vis_flow[:, :3]) * (x / 20))
    all_rays.append(ray_points)

all_rays = np.concatenate(all_rays)

vis_points = np.concatenate([points, all_rays])
vis_color = np.concatenate(
        [np.ones((points.shape[0], 4)) * (0,0,1,0.8),   # pts color blue
         np.ones((all_rays.shape[0], 4)) * (0,1,0,0.8)]    # flow color green
)


canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
view = canvas.central_widget.add_view()
canvas2 = vispy.scene.SceneCanvas(keys='interactive', show=True)
view2 = canvas2.central_widget.add_view()

interpolation = 'nearest'
img_data = (255 * np.random.normal(size=(100, 100, 3), scale=2, loc=128)).astype(np.ubyte)

fig, axes = plt.subplots(1,2, dpi=400)
axes[0].imshow(img_data)
axes[1].plot(range(10))
# fig.tight_layout()
fig.savefig(os.path.expanduser('~') + '/tmp/fig.png')

img_mpl = plt.imread(os.path.expanduser('~') + '/tmp/fig.png')


# arrows = np.concatenate([np.zeros(points.shape), points], axis=1)   # x,y,z,x2,y2,z2


# arr = Arrow(pos=pt, color='teal', method='gl', width=5., arrows=arrow,
#             arrow_type="angle_30", arrow_size=5.0, arrow_color='teal', antialias=True, parent=view.scene)



image = visuals.Image(img_mpl, interpolation=interpolation,
                            parent=view2.scene, method='subdivide')



view2.camera = vispy.scene.PanZoomCamera(aspect=1)
# flip y-axis to have correct aligment
view2.camera.flip = (0, 1, 0)
view2.camera.set_range()
view2.camera.zoom(1)

scatter = visuals.Markers()
symbols = np.random.choice(['o'], len(vis_points))
scatter.set_data(vis_points, edge_width=0, face_color=vis_color, size=5, symbol=symbols)

view.add(scatter)
view.camera = 'turntable'  # or try 'arcball'
axis = visuals.XYZAxis(parent=view.scene)

# vispy.app.run()
# canvas2 = vispy.scene.SceneCanvas(keys='interactive', show=True)
# view2 = canvas2.central_widget.add_view()
# generate data
# interpolation = 'nearest'
# img_data = (255 * np.random.normal(size=(100, 100, 3), scale=2, loc=128)).astype(np.ubyte)
#
# image = visuals.Image(img_data, interpolation=interpolation,
#                             parent=view2.scene, method='subdivide')
#
# view2.camera = vispy.scene.PanZoomCamera(aspect=1)
# # flip y-axis to have correct aligment
# view2.camera.flip = (0, 1, 0)
# view2.camera.set_range()
# view2.camera.zoom(1)
#

# import vispy.plot as vp
# from vispy import color
# from vispy.util.filter import gaussian_filter
# import numpy as np
#
# z = np.random.normal(size=(250, 250), scale=200)
# z[100, 100] += 50000
# z = gaussian_filter(z, (10, 10))
#
# fig = vp.Fig(show=False)
# cnorm = z / abs(np.amax(z))
# c = color.get_colormap("hsl").map(cnorm).reshape(z.shape + (-1,))
# c = c.flatten().tolist()
# c=list(map(lambda x,y,z,w:(x,y,z,w), c[0::4],c[1::4],c[2::4],c[3::4]))
#
# #p1 = fig[0, 0].surface(z, vertex_colors=c) # why doesn't vertex_colors=c work?
# p1 = fig[0, 0].surface(z)
# p1.mesh_data.set_vertex_colors(c)
# fig.show()


# one could stop here for the data generation, the rest is just to make the
# data look more interesting. Copied over from magnify.py
# centers = np.random.normal(size=(50, 3))
# indexes = np.random.normal(size=100000, loc=centers.shape[0] / 2,
#                            scale=centers.shape[0] / 3)
# indexes = np.clip(indexes, 0, centers.shape[0] - 1).astype(int)

# scales = 10**(np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
# pos *= scales
# pos += centers[indexes]

# create scatter object and fill in the data


# view2.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation



if __name__ == '__main__':
    import sys

    if sys.flags.interactive != 1:
        vispy.app.run()

