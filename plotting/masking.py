import numpy as np
import scipy.ndimage as nd


def fill_nearest(mask, data):
    ind = nd.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return data[tuple(ind)]


def mask_anomalous_edges(xx, yy, m=3):
    total_mask = np.zeros_like(xx, dtype=np.bool)
    dx = np.diff(xx, axis=1)
    dy = np.diff(yy, axis=0)
    x_mask = abs(dx - np.mean(dx)) > m * np.std(dx)
    x_mask &= abs(dx) > abs(np.mean(dx))
    y_mask = abs(dy - np.mean(dy)) > m * np.std(dy)
    y_mask &= abs(dy) > abs(np.mean(dy))
    total_mask[:-1, :] |= y_mask
    total_mask[1:, :] |= y_mask
    total_mask[:, :-1] |= x_mask
    total_mask[:, 1:] |= x_mask

    dx = np.diff(xx, axis=0)
    dy = np.diff(yy, axis=1)
    x_mask = abs(dx - np.mean(dx)) > m * np.std(dx)
    x_mask &= abs(dx) > abs(np.mean(dx))
    y_mask = abs(dy - np.mean(dy)) > m * np.std(dy)
    y_mask &= abs(dy) > abs(np.mean(dy))
    total_mask[:-1, :] |= x_mask
    total_mask[1:, :] |= x_mask
    total_mask[:, :-1] |= y_mask
    total_mask[:, 1:] |= y_mask
    return total_mask

def mask_not_finite(xx, yy):
    return (~np.isfinite(xx)) | (~np.isfinite(yy))

def edge_mask_2_center_mask(edge_mask):
    return edge_mask[:-1, :-1] | edge_mask[-1:, 1:] | edge_mask[1:, :-1] |  edge_mask[1:, 1:]