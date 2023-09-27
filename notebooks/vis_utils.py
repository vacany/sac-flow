#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# from vis.deprecated_vis import *
# generate instance and flow, show next to each other

pc = torch.randn(1,100)


from IPython import display

import matplotlib as mpl
fig, ax = plt.subplots(3, figsize=(10,10), edgecolor='w')
ax[0].axis('equal')
ax[1].axis('equal')
# ax[0].set_xlim(-3,3)
# ax[0].set_ylim(-3,3)

# ax[1].set_xlim(-3,3)
# ax[1].set_ylim(-3,3)

cmap = plt.cm.jet  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (.5, .5, .5, 1.0)
# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

# zaver je plotit to do slozky? a do 3D visualizeru? 
# 2d ploty lossu a metric v bunce?

N=1000

for i in range(10):
    ax[0].clear()
    ax[1].clear()
    
    P1 = np.random.randn(N, 3) * 4
    P2 = P1 + np.random.randn(N, 3) * 0.5
    instance = np.random.randint(1,10, N)
    flow = np.random.randn(N,3)
    cmap = mpl.cm.get_cmap('PiYG', 20) 
    
    ax[0].plot(P1[:,0], P1[:,1], '.b')
    ax[0].quiver(P1[:,0], P1[:,1], flow[:,0], flow[:,1], color='g', units='xy', scale=1)
    ax[0].plot(P2[:,0], P2[:,1], '.r')
    
    ax[1].scatter(P1[:,0], P1[:,2], c=instance, cmap=cmap)
    
    
    display.display(plt.gcf())
    display.clear_output(wait=True)
    
    # from mpl_toolkits.mplot3d import Axes3D
    # ax[2] = Axes3D(fig)
    # ax[2].scatter(P1[:,0], P1[:,1], P2[:,2], c=instance, cmap=cmap)

# add exclusivity loss, upravit kody a pripravit pro github, experimenty etc.
