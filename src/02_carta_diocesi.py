#!/usr/bin/env python3
# coding: utf-8
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2023 Federico Motta    <federico.motta@unimore.it>
#                    Lorenzo  Carletti <lorenzo.carletti@unimore.it>
#                    Matteo   Vanzini  <matteo.vanzini@unimore.it>
#                    Andrea   Serafini <andrea.serafini@unimore.it>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Build a dendrogram by running UPGMA over some language parameters

UPGMA (Unweighted Pair Group Method with Arithmetic mean) is a
simple agglomerative (bottom-up) hierarchical clustering method

Usage:
     mkdir -p out_plot && python3 src/01_plot_clusters.py -i out_preprocess -o out_plot | grep -i record | grep -o '\\[.*\\]' | sort -V |uniq -c
"""

from matplotlib import colormaps
from shapely.geometry import Point
from utility import get_cli_parser, A4_PORTRAIT_PAGE_SIZE_INCHES
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

def color_rgb_tuple_to_hex(color_input):
    return '#{:02x}{:02x}{:02x}'.format(int(color_input[0] * 255), int(color_input[1] * 255), int(color_input[2] * 255)) 


DPI = 300
filler_color = "#EEEEEE"
border_color = "#000000"

parser = get_cli_parser(__doc__, __file__)
parser.add_argument(
    "-i",
    "--input_file",
    default="./clusters.txt",
    help="File containing clusters given by 01_plot_clusters.py",
    metavar="str",
    required=True,
    type=str,
)
parser.add_argument(
    "-m",
    "--map_file",
    default="shapefile_diocese/diocese1250.shp",
    help="File containing the shape file to use for the plots",
    metavar="str",
    required=False,
    type=str,
)
parser.add_argument(
    "-o",
    "--output_dir",
    default=".",
    help="Directory where to put the resulting plots",
    metavar="str",
    required=True,
    type=str,
)
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    dest="verbosity",
)

parsed_args = parser.parse_args()

df = pd.read_excel("datasets/99_dialects_lat_long_geolocation_clean.xlsx")

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

lat, lon = df.Latitude, df.Longitude
geometry = [Point(xy) for xy in zip(lon, lat)]

clusters = list()
is_last_clusterless = False
title = ""
with open(parsed_args.input_file, "r") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if i == 0:
            title = line
            continue
        if not line:
            continue
        if line == "C:":
            is_last_clusterless = True
            continue

        line = line.split(" ")
        clusters.append(line)

colors = colormaps.get_cmap("jet")

italy = gpd.read_file(parsed_args.map_file)
italy.geometry = italy.geometry.make_valid()

geo_df = gpd.GeoDataFrame(geometry=geometry)
geo_df = geo_df.assign(
    lat=lambda _df: _df["geometry"].apply(lambda pt: pt.y),
    lon=lambda _df: _df["geometry"].apply(lambda pt: pt.x),
    values=[None for i in range(geo_df.shape[0])],
    color=[None for i in range(geo_df.shape[0])],
)

legend_handles = dict()

for i, cluster in enumerate(clusters):
    for lat, lon in df.loc[
        df.Label.isin(cluster), ["Latitude", "Longitude"]
    ].itertuples(index=False):
        value = float(i / len(clusters))
        color_str = color_rgb_tuple_to_hex(colors(value))
        geo_df.loc[geo_df.lat.eq(lat) & geo_df.lon.eq(lon), "values"] = value
        geo_df.loc[geo_df.lat.eq(lat) & geo_df.lon.eq(lon), "color"] = color_str
        chosen_label = str(i)
        if i >= (len(clusters) - 1) and is_last_clusterless:
            chosen_label = "No Cluster"
        geo_df.loc[
            geo_df.lat.eq(lat) & geo_df.lon.eq(lon),
            "label",
        ] = chosen_label
        # Basically, prepare the legend's stuff
        legend_handles[value] = plt.plot([],color=color_str, ms=9, mec="none",
                        label=chosen_label, ls="", marker="o")[0]

# Kill off the plot used to prepare the legend
fig = plt.gcf()
plt.clf()
plt.close(fig)

italy.crs = {"init": "epsg:4326"}
geo_df.crs = {"init": "epsg:4326"}

# Set color to fill_color by default for each polygon
italy = italy.assign(color = [filler_color] * italy.shape[0])
# Get the polygon ID that the clusters' positions match.
# Replace the color there...
merge_result = italy.sjoin(geo_df, how='inner', predicate='contains')
for i, row in merge_result.iterrows():
    # None check needed as geo_df contains some regions without languages...
    if row['color_right'] is not None:
        italy.at[i, 'color'] = row['color_right']

ax = italy.plot(alpha=1.0, color=italy.color, zorder=1)
ax = gpd.GeoSeries(
    italy.to_crs(epsg=4326)["geometry"].unary_union,
).boundary.plot(ax=ax, alpha=0.5, color=border_color, zorder=2, lw=0.5)

final_legend_handles = []
for elem in sorted(legend_handles.keys()):
    final_legend_handles += [legend_handles[elem]]

plt.legend(loc="upper right", handles=final_legend_handles)
plt.title(title)
fig = plt.gcf()
fig.set_size_inches(A4_PORTRAIT_PAGE_SIZE_INCHES)
fig.tight_layout()  # hai modificato FrM e la lat lon di taranto nel
# 99_clean.xlsx
fig.savefig(parsed_args.input_file.replace(".txt", "__mappa.svg"), dpi=600)
plt.show()
