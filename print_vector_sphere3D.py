# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:55:32 2021

@author: hiper
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

#fig1,ax1 = plt.subplots( dpi=600)
fig = plt.figure(dpi=600)
ax1 = fig.gca(projection='3d')
#ax1.set_aspect("equal")

'''
# draw cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax1.plot3D(*zip(s, e), color="b")
'''
# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax1.plot_wireframe(x, y, z, color="r")
'''
# draw a point
ax.scatter([0], [0], [0], color="g", s=100)
'''
# draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


xx,yy,zz=.919,-.358,.162

a = Arrow3D([0, xx], [0, yy], [0, zz], mutation_scale=20,
            lw=1, arrowstyle="-|>", color="k")
ax1.add_artist(a)
plt.show()
