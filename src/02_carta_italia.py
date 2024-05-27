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
        mkdir -p out_plot && python3 src/01_plot_clusters.py -i out_preprocess -o out_plot | grep -i record | grep -o '\[.*\]' | sort -V |uniq -c
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from matplotlib import colormaps
from utility import get_cli_parser

# ---- MACRO ----

DPI         = 300

# -- END MACRO --

# ---- START ARGDEF ----

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
    "-o",
    "--output_dir",
    default=".",
    help="Directory where to put the resulting plots",
    metavar="str",
    required=True,
    type=str,
)

parsed_args = parser.parse_args()

# ----- END ARGDEF -----

df = pd.read_excel(
    "datasets/99_dialects_lat_long_geolocation_clean.xlsx"
)

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

lat = df.Latitude
long = df.Longitude
geometry = [Point(xy) for xy in zip(long, lat)]
is_last_clusterless = False

clusters = list()
num_line = 0
title = ""
with open(parsed_args.input_file, "r") as f:
    for line in f:
        line = line.strip()
        num_line += 1
        if num_line == 1:
            title = line
            continue
        if len(line) <= 0:
            continue
        if line == "C:":
            is_last_clusterless = True
            continue

        line = line.split(" ")
        clusters.append(line)

colors = colormaps.get_cmap("jet")

markers = ["o", "^", "s", "*", "+", "x", "D", "v", "<", ">"]
markers = markers[: len(clusters) + 1]

italy = gpd.read_file("src/italy/ITA_adm3.shp")


geo_df = gpd.GeoDataFrame(geometry=geometry)
geo_df = geo_df.assign(
    lat=lambda _df: _df["geometry"].apply(lambda pt: pt.y),
    long=lambda _df: _df["geometry"].apply(lambda pt: pt.x),
    marker=[None for i in range(geo_df.shape[0])],
    values=[None for i in range(geo_df.shape[0])],
)

for i, cluster in enumerate(clusters):
    marker = markers[i]

    for lat, long in df.loc[
        df.Label.isin(cluster), ["Latitude", "Longitude"]
    ].itertuples(index=False):
        geo_df.loc[
            geo_df.lat.eq(lat) & geo_df.long.eq(long), "marker"
        ] = marker
        geo_df.loc[
            geo_df.lat.eq(lat) & geo_df.long.eq(long), "values"
        ] = i / len(clusters)
        chosen_label = str(i)
        if i >= (len(clusters) - 1) and is_last_clusterless:
            chosen_label = "No Cluster"
        geo_df.loc[
            geo_df.lat.eq(lat) & geo_df.long.eq(long), "label"
        ] = chosen_label

italy.crs = {"init": "epsg:4326"}
geo_df.crs = {"init": "epsg:4326"}

ax = italy.plot(alpha=0.35, color="#3e9df0", zorder=1)
ax = gpd.GeoSeries(
    italy.to_crs(epsg=4326)["geometry"].unary_union
).boundary.plot(ax=ax, alpha=0.5, color="#0767ba", zorder=2, lw=0.5)

by = ["values", "marker", "label"]

for i, (idx, _df) in enumerate(geo_df.groupby(by, as_index=True)):
    plot_kwargs = dict(zip(by, idx))
    plot_kwargs["color"] = colors(plot_kwargs.pop("values"))

    ax = _df.plot(ax=ax, markersize=50, zorder=3, **plot_kwargs)

plt.legend(loc="upper right")
plt.title(title)
#plt.savefig(os.path.join(parsed_args.output_dir, title.replace(" ", "_") + "_italy_map.png"), dpi=DPI)
plt.show()
