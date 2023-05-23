import numpy as np
import mayavi.mlab as mlab

pc = np.load('/home/patrik/cmp/pcflow/pc.npy')
vel = np.load('/home/patrik/cmp/pcflow/velocity.npy')
mos = np.linalg.norm(vel, axis=1) > 0.05

fig = mlab.figure(1, bgcolor=(1,1,1))


mlab.points3d(pc[mos==False, 0], pc[mos==False, 1], pc[mos==False, 2], scale_factor=0.1, color=(0, 0, 1), figure=fig)
mlab.points3d(pc[mos, 0], pc[mos, 1], pc[mos, 2], scale_factor=0.1, color=(1, 0, 0), figure=fig)
mlab.quiver3d(pc[:, 0], pc[:, 1], pc[:, 2], vel[:, 0], vel[:, 1], vel[:, 2], scale_factor=1, color=(0, 1, 0), figure=fig)

mlab.show()
