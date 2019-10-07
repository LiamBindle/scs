import numpy as np
from gcpy.grid.horiz import csgrid_GMAO


def rotate_vectors(x, y, z, k, theta):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    v = np.moveaxis(np.array([x, y, z]), 0, -1)  # shape: (..., 3)
    v = v*np.cos(theta) + np.cross(k, v) * np.sin(theta) + k[np.newaxis, :] * np.dot(v, k)[:, np.newaxis] * (1-np.cos(theta))
    return v[..., 0], v[..., 1], v[..., 2]


def cartesian_to_spherical(x, y, z):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    # Calculate x,y in spherical coordinates
    y_sph = np.arcsin(z)
    x_sph = np.arctan2(y, x)
    return x_sph, y_sph


def spherical_to_cartesian(x, y):
    x_car = np.cos(y) * np.cos(x)
    y_car = np.cos(y) * np.sin(x)
    z_car = np.sin(y)
    return x_car, y_car, z_car


def schmidt_transform(x, y, s):
    D = (1 - s ** 2) / (1 + s ** 2)
    y = np.arcsin((D + np.sin(y)) / (1 + D * np.sin(y)))
    return x, y


def scs_transform(x, y, s, tx, ty):
    # Convert xy to radians
    x = x * np.pi / 180
    y = y * np.pi / 180
    tx = tx * np.pi / 180
    ty = ty * np.pi / 180
    # Calculate rotation about x, and z axes
    x0 = np.pi
    y0 = -np.pi/2
    theta_x = ty - y0
    theta_z = tx - x0
    # Apply schmidt transform
    x, y = schmidt_transform(x, y, s)
    # Convert to cartesian coordinates
    x, y, z = spherical_to_cartesian(x, y)
    # Rotate about x axis
    xaxis = np.array([0, 1, 0])
    x, y, z = rotate_vectors(x, y, z, xaxis, theta_x)
    # Rotate about z axis
    zaxis = np.array([0, 0, 1])
    x, y, z = rotate_vectors(x, y, z, zaxis, theta_z)
    # Convert back to spherical coordinates
    x, y = cartesian_to_spherical(x, y, z)
    # Convert back to degrees and return
    x = x * 180 / np.pi
    y = y * 180 / np.pi
    return x, y

def make_grid_SCS(csres: int, stretch_factor: float, target_lat: float, target_lon: float):
    csgrid = csgrid_GMAO(csres, offset=0)
    csgrid_list = [None]*6
    for i in range(6):
        lat = csgrid['lat'][i].flatten()
        lon = csgrid['lon'][i].flatten()
        lon, lat = scs_transform(lon, lat, stretch_factor, target_lon, target_lat)
        lat = lat.reshape((csres, csres))
        lon = lon.reshape((csres, csres))
        lat_b = csgrid['lat_b'][i].flatten()
        lon_b = csgrid['lon_b'][i].flatten()
        lon_b, lat_b = scs_transform(lon_b, lat_b, stretch_factor, target_lon, target_lat)
        lat_b = lat_b.reshape((csres+1, csres+1))
        lon_b = lon_b.reshape((csres+1, csres+1))
        csgrid_list[i] = {'lat': lat,
                          'lon': lon,
                          'lat_b': lat_b,
                          'lon_b': lon_b}
    return [csgrid, csgrid_list]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import plotting.cs
    csgrid, csgrid_list = make_grid_SCS(24, 1.8, 40, 265)
    plt.figure()
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    ax.stock_img()
    ax.set_global()
    ax.coastlines(linewidth=0.6, alpha=0.5)
    for i in range(6):
        plotting.cs.plot_gridboxes(csgrid_list[i]['lon_b'],csgrid_list[i]['lat_b'], ax, proj)
        plotting.cs.plot_facenumber(csgrid_list[i]['lon_b'],csgrid_list[i]['lat_b'], i+1, ax, proj)
    plt.tight_layout()
    plt.show()
    print('Done')

    # x = np.array([0])
    # y = np.array([-90])
    # print(scs_transform(x, y, 1.8, 265, 40))

    # x, y, z = spherical_to_cartesian(45*np.pi/180, 45*np.pi/180)
    # k = np.array([0, 0, 1])
    # theta = np.pi/2
    # x2, y2, z2 = rotate_vectors(x, y, z, k, theta)
    # x1_lat, y1_lat = cartesian_to_spherical(x, y, z)
    # x2_lat, y2_lat = cartesian_to_spherical(x2, y2, z2)
    # print(f'initial: ({x1_lat[0]*180/np.pi}, {y1_lat[0]*180/np.pi})')
    # print(f'final:   ({x2_lat[0] * 180 / np.pi}, {y2_lat[0] * 180 / np.pi})')
