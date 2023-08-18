#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import sys
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point


# In[39]:


df = pd.read_excel(
    "datasets/99_dialects_lat_long_geolocation_clean.xlsx"
)  # .drop(columns="Unnamed: 0")
df.columns = [
    "Language",
    "Label",
    "Glottocode",
    "Iso 639-3 Code",
    "Dialect Group",
    "Location",
    "Latitude",
    "Longitude",
]
# df


# In[3]:


lat = df.Latitude
long = df.Longitude
geometry = [Point(xy) for xy in zip(long, lat)]


# In[26]:


# 4 : ['BA', 'Bar', 'Pca', 'SMC']
# 10 : ['Bot', 'CZ', 'Cel', 'Mus', 'RC', 'RG', 'Rib', 'SFM', 'SMX', 'TP']
# 5 : ['Cas', 'Cor', 'Nov', 'PR', 'RE']
# 4 : ['Cut', 'Fel', 'Mes', 'Nic']
# 2 : ['Gar', 'Ver']
clusters = [
    ["BA", "Bar", "PCa", "SMC"],
    ["Bot", "CZ", "Cel", "Mus", "RC", "RG", "Rib", "SFM", "SMX", "TP"],
    ["Cas", "Cor", "Nov", "PR", "RE"],
    ["Cut", "Fel", "Mes", "Nic"],
    ["Gar", "Ver"],
]
# non_clusterised = df
# for cluster in clusters:
#     non_clusterised = non_clusterised.loc[~non_clusterised.Label.isin(cluster),:]

# clusters.append(non_clusterised.Label.to_list())


# In[31]:


from matplotlib import colormaps

colors = colormaps.get_cmap("jet")

markers = ["o", "^", "s", "*", "+", "x", "D"]
markers = markers[: len(clusters) + 1]


# In[ ]:


italy = gpd.read_file("src/italy/gadm41_ITA_1.shp")


# In[56]:


geo_df = gpd.GeoDataFrame(geometry=geometry)
geo_df = geo_df.assign(
    lat=lambda _df: _df["geometry"].apply(lambda pt: pt.y),
    long=lambda _df: _df["geometry"].apply(lambda pt: pt.x),
    marker=[None for i in range(geo_df.shape[0])],
    values=[None for i in range(geo_df.shape[0])],
)

for i, cluster in enumerate(clusters):
    marker = markers[i]
    # color = colors(i/len(clusters))

    for lat, long in df.loc[
        df.Label.isin(cluster), ["Latitude", "Longitude"]
    ].itertuples(index=False):
        geo_df.loc[
            geo_df.lat.eq(lat) & geo_df.long.eq(long), "marker"
        ] = marker
        geo_df.loc[
            geo_df.lat.eq(lat) & geo_df.long.eq(long), "values"
        ] = i / len(clusters)

# geo_df


# In[57]:


italy.crs = {"init": "epsg:4326"}
geo_df.crs = {"init": "epsg:4326"}


# In[66]:


ax = italy.plot(alpha=0.35, color="#3e9df0", zorder=1)
ax = gpd.GeoSeries(
    italy.to_crs(epsg=4326)["geometry"].unary_union
).boundary.plot(ax=ax, alpha=0.5, color="#0767ba", zorder=2, lw=0.5)

by = ["marker", "values"]

for i, (idx, _df) in enumerate(geo_df.groupby(by, as_index=True)):
    plot_kwargs = dict(zip(by, idx))
    plot_kwargs["color"] = colors(plot_kwargs.pop("values"))

    ax = _df.plot(ax=ax, markersize=50, zorder=3, label=i, **plot_kwargs)

plt.legend(loc="upper right")
plt.show()
