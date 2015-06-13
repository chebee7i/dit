"""
Project and visualize a simplex grid on the 3-simplex.

"""
from __future__ import division

import dit
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simplex3_vertices():
    """
    Returns the vertices of the standard 3-simplex. Each column is a vertex.

    """
    v = np.array([
        [1, 0, 0],
        [-1/3, +np.sqrt(8)/3, 0],
        [-1/3, -np.sqrt(2)/3, +np.sqrt(2/3)],
        [-1/3, -np.sqrt(2)/3, -np.sqrt(2/3)],
    ])
    return v.transpose()

def plot_simplex_vertices(ax=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')

    vertices = simplex3_vertices()
    lines = np.array([
        vertices[:,0],
        vertices[:,1],
        vertices[:,2],
        vertices[:,0],
        vertices[:,3],
        vertices[:,1],
        vertices[:,3],
        vertices[:,2]
    ])
    lines = lines.transpose()
    out = ax.plot(lines[0], lines[1], lines[2])
    ax.set_axis_off()
    ax.set_aspect('equal')
    return ax, out, vertices

def plot_simplex_grid(subdivisions, ax=None):

    ax, out, vertices = plot_simplex_vertices(ax=ax)

    grid = dit.simplex_grid(4, subdivisions)
    projections = []
    for dist in grid:
        pmf = dist.pmf
        proj = (pmf * vertices).sum(axis=1)
        projections.append(proj)

    projections = np.array(projections)
    projections = projections.transpose()
    out = ax.scatter(projections[0], projections[1], projections[2], s=10)
    return ax, out, projections

if __name__ == '__main__':
    plot_simplex_grid(9)
    plt.show()
