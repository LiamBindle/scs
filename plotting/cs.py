import pyproj
import numpy as np

from plotting.masking import *


def plot_gridboxes(xe, ye, ax, proj, xeye_crs = pyproj.Proj(init='epsg:4326'), mask_std=3):
    xx, yy = pyproj.transform(xeye_crs, pyproj.Proj(proj.proj4_init), xe, ye)
    mask = mask_anomalous_edges(xx, yy, m=mask_std)
    xx = np.ma.masked_where(mask, xx)
    yy = np.ma.masked_where(mask, yy)
    for x, y in zip(xx, yy):
        ax.plot(x, y, color='black', linewidth=0.6, alpha=0.4)
    for x, y in zip(xx.transpose(), yy.transpose()):
        ax.plot(x, y, color='black', linewidth=0.6, alpha=0.4)
    ax.plot(xx[0,:], yy[0,:], color='black', alpha=0.2)
    ax.plot(xx[-1, :], yy[-1, :], color='black', alpha=0.2)
    ax.plot(xx[:, 0], yy[:, 0], color='black', alpha=0.2)
    ax.plot(xx[:, -1], yy[:, -1], color='black', alpha=0.2)


def plot_data(xe, ye, data, ax, proj, xeye_crs = pyproj.Proj(init='epsg:4326'), mask_std=3, **kwargs):
    xx, yy = pyproj.transform(xeye_crs, pyproj.Proj(proj.proj4_init), xe, ye)
    # Mask hidden edges
    mask = mask_not_finite(xx, yy)
    xx = fill_nearest(mask, xx)
    yy = fill_nearest(mask, yy)
    # Mask edges wrapping projection edges
    mask = mask_anomalous_edges(xx, yy, m=mask_std)
    xx = fill_nearest(mask, xx)
    yy = fill_nearest(mask, yy)
    # Mask remaining bad values
    mask2 = mask_not_finite(xx, yy)
    xx[mask2] = 0
    yy[mask2] = 0
    mask = edge_mask_2_center_mask(mask)
    data = np.ma.masked_where(mask, data)
    pc = ax.pcolormesh(xx, yy, data, **kwargs)

def plot_facenumber(xe, ye, face, ax, proj, xeye_crs = pyproj.Proj(init='epsg:4326')):
    xx, yy = pyproj.transform(xeye_crs, pyproj.Proj(proj.proj4_init), xe, ye)
    middle = np.array(xx.shape, dtype=np.int) // 2
    ax.text(xx[middle[0], middle[1]], yy[middle[0], middle[1]], f'{face}')

