from orbit_predictor.sources import NoradTLESource

source = NoradTLESource.from_url("http://www.celestrak.com/NORAD/elements/resource.txt")
predictor = source.get_predictor("NUSAT-3")
from orbit_predictor.locations import Location

FBR = Location("Observatori Fabra", 41.4184, 2.1239, 408)
predicted_pass = predictor.get_next_pass(FBR)
predicted_pass
predicted_pass.aos  # Acquisition Of Signal
predicted_pass.duration_s
predictor.get_position(predicted_pass.aos).position_llh

import matplotlib.pyplot as plt
plt.ion()

import cartopy.crs as ccrs

import pandas as pd
dates = pd.date_range(start="2017-12-11 00:00", periods=1000, freq="30S")
latlon = pd.DataFrame(index=dates, columns=["lat", "lon"])

for date in dates:
    lat, lon, _ = predictor.get_position(date).position_llh
    latlon.loc[date] = (lat, lon)

plt.figure(figsize=(15, 25))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()

plt.plot(latlon["lon"], latlon["lat"], 'k',
         transform=ccrs.Geodetic());
plt.show()