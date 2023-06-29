# ARGOVERSE
# https://www.argoverse.org/av1.html
# https://argoverse.github.io/user-guide/datasets/sensor.html
# https://arxiv.org/pdf/1911.02620.pdf

#                                                       t_x       t_y       t_z
#  up_lidar  0.999996  0.000000  0.000000 -0.002848  1.350180  0.000000  1.640420
#  down_lidar -0.000089 -0.994497  0.104767  0.000243  1.355162  0.000133  1.565252

# not in lidar frame here!
# ---> to shift to right coordinate system for pitch and yaw

t_x1 = 1.350180
t_x2 = 1.355162
t_x_mean = (t_x1 + t_x2) / 2

t_y1 = 0.000000
t_y2 = 0.000133
t_y_mean = (t_y1 + t_y2) / 2

t_z1 = 1.640420
t_z2 = 1.565252
t_z_mean = (t_z1 + t_z2) / 2

lidar_pose = (t_x_mean, t_y_mean, t_z_mean)

fov_up= 25
fov_down = -25
proj_H = 50 #?
proj_W = 2048

data_config = {'lidar_pose' : lidar_pose,
               'fov_up' : fov_up,
               'fov_down' : fov_down,
               'proj_H' : proj_H,
               'proj_W' : proj_W
               }
