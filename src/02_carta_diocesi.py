#!/usr/bin/env python3
# coding: utf-8
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2025 Federico Motta    <federico.motta@unimore.it>
#                         Lorenzo  Carletti <lorenzo.carletti@unimore.it>
#
# Copyright (C)      2023 Federico Motta    <federico.motta@unimore.it>
#                         Lorenzo  Carletti <lorenzo.carletti@unimore.it>
#                         Matteo   Vanzini  <matteo.vanzini@unimore.it>
#                         Andrea   Serafini <andrea.serafini@unimore.it>
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
r"""
Geovisualize clusters with a thematic map

Usage:
    export WORKSPACE="out_plot";              \
    mkdir -p "${PLOT_DIR}";                   \
    && python3 src/02_carta_diocesi.py        \
        -i "${WORKSPACE}/clusters.txt         \
        -m "${WORKSPACE}/new_diocesi_1250.shp"     \
        -o "${WORKSPACE}/map_clusters.pdf"
"""

from argparse import FileType
from functools import cache
from logging import debug, info  # , warning, critical
from matplotlib import colormaps
from os.path import basename
from preprocessing import clean_excel_file
from shapely.geometry import Point
from utility import (
    A4_PORTRAIT_PAGE_SIZE_INCHES,
    get_cli_parser,
    initialize_logging,
)
import geopandas as gpd
import matplotlib.pyplot as plt


__DEFAULT = dict(
    allowed_extensions=("jpg", "pdf", "png", "svg"),
    border_color="#0767ba",
    dpi=300,
    fill_color="#bfdff9",  # aka rgba="#3e9df059" to rgb according to gimp
    geo_coordinates="./datasets/99_dialects_lat_long_geolocation_clean.xlsx",
    input_file="./out_plot/clusters.txt",
    map_file="./shapefile_diocese/new_diocesi_1250.shp",
    output_file="./out_plot/mappa_clusters.pdf",
    do_markers=False,
    do_areas=True,
    verbose=0,
)


def component2hex(f):
    assert f >= 0 and f <= 1, f"{f=} shoult be a float in [0, 1]"
    return f"{int(f * 255):02x}"


def dms2dd(degrees, minutes, seconds, direction):
    """Convert Degrees, Minutes, and Seconds (DMS) to Decimal Degrees (DD)"""
    assert direction in ("N", "S", "E", "W"), repr(direction)
    degrees, minutes, seconds = tuple(
        map(lambda v: abs(int(v)), (degrees, minutes, seconds))
    )
    coeff = -1 if direction in ["W", "S"] else 1
    ret = coeff * sum((degrees, minutes / 60.0, seconds / 3600.0))
    debug(f"{(degrees, minutes, seconds)=}\t{direction} = {ret: >+10.6f}")
    return ret


def get_final_figure(
    legend_handles,
    title,
    size=A4_PORTRAIT_PAGE_SIZE_INCHES,
    show=False,
    tight_layout=True,
):
    plt.legend(
        loc="upper right",
        bbox_to_anchor=(0.95, 0.95),
        handles=legend_handles,
    )
    plt.title(title)
    fig = plt.gcf()  # do not change the order of the following operations
    if show:
        plt.show()
    fig.set_size_inches(size)
    if tight_layout:
        fig.tight_layout()
    return fig


def get_geo_filled_map(
    shp_df,
    ax=None,
    color_col="color",
    alpha=0.5,
    zorder=1,
    **kwargs,
):
    return shp_df.plot(
        alpha=alpha,
        ax=ax,
        color=shp_df[color_col],
        zorder=zorder,
        **kwargs,
    )


def get_legend_handles(
    clusters, df, geo_df, last_is_clusterless=False, legend_kwargs=None
):
    markers = ["o", "^", "s", "D", "v", "<", ">", "*", "+", "x"]
    markers = markers[: len(clusters) + 1]
    examined_languages = set()
    if legend_kwargs is None:
        legend_kwargs = dict(ms=9, mec="none", ls="", marker="o")
    legend_handles = dict()
    for i, cluster in enumerate(clusters):
        marker = markers[i]
        label = str(i)
        if last_is_clusterless and i + 1 >= len(clusters):
            label = "No Cluster"
        info(f"Cluster {label:>10s} = {cluster!s}")
        value = float(i / len(clusters))
        color = link_color_func(value)
        debug(f"cluster={label}\t{value=}\t{color=}")
        for lat, lon in df.loc[
            df.index.isin(cluster), ["Latitude", "Longitude"]
        ].itertuples(index=False):
            selector = geo_df["latitude"].eq(lat) & geo_df["longitude"].eq(lon)
            geo_df.loc[selector, "color"] = color
            geo_df.loc[selector, "lan_name"] = df.index[selector == True][0]
            examined_languages.add(df.index[selector == True][0])
            geo_df.loc[selector, "label"] = label
            geo_df.loc[selector, "values"] = value
            geo_df.loc[selector, "marker"] = marker
            legend_kwargs["marker"] = marker
            legend_handles[value] = plt.plot(
                list(), color=color, label=label, **legend_kwargs
            ).pop(0)
    missing_languages = set()
    for cluster in clusters:
        for elem in cluster:
            if elem not in examined_languages:
                missing_languages.add(elem)
    if len(missing_languages) > 0:
        info(f"Languages missing point in map:\n{missing_languages}")
    debug(f"geo_df=\n\t{geo_df.to_string()}")
    geo_df = geo_df.loc[geo_df["label"].notna(), :]
    debug(f"geo_df=\t#dropped cities without cluster\n{geo_df.to_string()}")
    legend_handles = [
        v
        for k, v in sorted(
            legend_handles.items(),
            key=lambda t: t[0],
        )
    ]
    debug(f"{legend_handles=}")

    plt.clf()  # clear the current figure
    plt.cla()  # and axes
    return legend_handles, geo_df.copy(deep=True).sort_index()


def get_map_border(
    shp_df,
    ax=None,
    alpha=0.5,
    color="#000000",
    crs="epsg:4326",
    zorder=2,
    **kwargs,
):
    assert "geometry" in shp_df.columns
    return gpd.GeoSeries(
        shp_df["geometry"].make_valid().union_all(), crs=crs
    ).boundary.plot(ax=ax, alpha=alpha, color=color, zorder=zorder, **kwargs)


@cache  # just removes redundant debugging messages
def link_color_func(value):
    return rgb2hex(colormaps.get_cmap("jet")(value))


def load_clusters(io_obj):
    title, clusters, last_is_clusterless = "", list(), False
    for i, line in enumerate(io_obj.readlines()):
        line = line.strip()
        if not line or i == 0:
            title = line
            continue
        if line.startswith("C:"):
            last_is_clusterless = True
            continue
        clusters.append(line.split(" "))
    info(f"Title = {title}")
    return title, clusters, last_is_clusterless


def load_lat_lon_file(**kwargs):
    crs = kwargs.pop("crs", "epsg:4326")

    df = clean_excel_file(**kwargs)
    df = (
        df.rename(
            columns={
                next(
                    (c for c in df.columns if c.lower().startswith("label")),
                    "Label",  # fallback value has no trailing obscenities
                ): "Label"
            }
        )
        .loc[:, ["Label", "Latitude", "Longitude"]]
        .sort_values("Label")
        .set_index("Label")
        .rename_axis("Language", axis=0)
    )
    debug(f"geo_coordinates=\n{df.to_string()}")
    df = df.astype("float")
    debug(f"geo_coordinates=\n{df.to_string()}")

    geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
    debug(
        "geometry=\t# https://geopandas.org/en/stable/docs/reference/api/"
        f"geopandas.GeoDataFrame.set_geometry.html\n{geometry!r}"
    )

    geo_df = gpd.GeoDataFrame(crs=crs, geometry=geometry)
    debug(f"geo_df(\t{crs=}\t)=\n{geo_df.to_string()}")
    geo_df = geo_df.assign(
        latitude=lambda _df: _df["geometry"].apply(lambda pt: pt.y),
        longitude=lambda _df: _df["geometry"].apply(lambda pt: pt.x),
        values=[None for i in range(geo_df.shape[0])],
        color=[None for i in range(geo_df.shape[0])],
    )
    debug(f"geo_df=\n{geo_df.to_string()}")
    return df, geo_df.sort_index()


def load_shapefile(
    filename,
    clip_box=None,
    crs="epsg:4326",
    fill_alpha=0.35,
    fill_color="#ffffff",
):
    """
    https://geopandas.org/en/stable/docs/reference/api/geopandas.read_file.html
    https://it.wikipedia.org/wiki/Italia_(regione_geografica)#Punti_estremi
    north = 47° 04′ 20″ N
    south = 35° 47′ 04″ N
    east  = 18° 31′ 13″ E
    west  = 06° 32′ 52″ E
    Edited east to 19° 50' to account for Cyprus
    """
    shp_df = gpd.read_file(filename)
    shp_df.geometry = shp_df.geometry.make_valid()
    shp_df.set_crs(crs)
    if clip_box is not None and not isinstance(clip_box, (list, tuple)):
        assert clip_box == "italy", f"Please implement {clip_box=}"
        clip_box = nsew2xy(
            **dict(
                north="47 04 20 N",
                south="35 47 04 N",
                east=" 19 50 00 E",
                west=" 06 32 52 E",
            )
        )
        debug(f"{clip_box=}")
    shp_df = shp_df.clip(clip_box)
    shp_df = shp_df.assign(
        # fill by default each polygon in the shapefile with fill_color
        color=lambda _: fill_color
    )
    return shp_df


def nsew2xy(**kwargs):
    "North, south, east, west  to  minx, miny, maxx, maxy" ""
    debug(f"{kwargs!r}")
    n, s, e, w = [kwargs.pop(k) for k in "north south east west".split()]
    debug(f"{(n, s, e, w)=}")
    if isinstance(n, str) and len(n.split()) == 4:
        n, s, e, w = [dms2dd(*coor.split()) for coor in (n, s, e, w)]
    debug(f"{(n, s, e, w)=}")
    assert s < n, f"{s=}\t{n=}"
    assert w < e, f"{w=}\t{e=}"
    return (w, s, e, n)


def rgb2hex(color):
    debug(f"rgb={color}")
    assert isinstance(color, (list, tuple)) and len(color) in {3, 4}
    ret = "#" + "".join(map(component2hex, color))
    debug(f"hex={ret}")
    return ret


def boolean_value(s):
    try:
        return int(s) != 0
    except ValueError:
        pass

    lower_s = s.lower()
    if lower_s not in {"false", "true"}:
        raise ValueError("Not a valid boolean value")
    return lower_s == "true"


def thematic_map(**kwargs):
    for k, v in kwargs.items():
        assert k in __DEFAULT, str(
            "Unrecognized parameter"
            f" {thematic_map.__name__}( ... {k}={v!r} ... )"
            f"\nPlease use only the available ones:\n\t"
            + "\n\t".join(sorted(__DEFAULT.keys(), key=str.lower))
        )
    for k, v in __DEFAULT.items():
        if k not in kwargs:
            kwargs[k] = v
    debug(f"\n\n\nCalling {thematic_map.__name__}(")
    for k in sorted(kwargs.keys(), key=str.lower):
        v = kwargs.pop(k)
        if (hasattr(v, "mode") and "b" in v.mode) and (
            hasattr(v, "name") and "std" not in v.name and ".xls" not in v.name
        ):  # avoid race conditions on file closure which may truncate it
            v.close()
            v = v.name
        kwargs[k] = v
        debug(f"{k:<16} = {v!r}")
    debug(")\n\n\n")

    df, geo_df = load_lat_lon_file(
        input=kwargs["geo_coordinates"],
        maintain_original_values=True,
        pivot_cell="Label",  # this trailing char is amazingly obscene
        pivot_cell_type="feature",
        sort_rows=False,
        sort_columns=True,
    )

    shp_df = load_shapefile(
        kwargs["map_file"], clip_box="italy", fill_color=kwargs["fill_color"]
    )

    title, clusters, last_is_clusterless = load_clusters(kwargs["input_file"])
    legend_handles, geo_df = get_legend_handles(
        clusters, df, geo_df, last_is_clusterless
    )

    if kwargs["do_areas"]:
        shp_df = update_left_with_merged_info(left=shp_df, right=geo_df)

    color_alpha = 1.0
    if kwargs["do_markers"] and kwargs["do_areas"]:
        color_alpha = 0.5

    ax = get_geo_filled_map(shp_df, alpha=color_alpha)
    ax = get_map_border(shp_df, ax=ax, color=kwargs["border_color"], lw=0.5)

    if kwargs["do_markers"]:
        by = ["marker", "label", "color"]
        for i, (idx, _df) in enumerate(geo_df.groupby(by, as_index=True)):
            plot_kwargs = dict(zip(by, idx))
            ax = _df.plot(ax=ax, markersize=50, zorder=3, **plot_kwargs)

    fig = get_final_figure(legend_handles, title, show=not kwargs["verbose"])
    # you changed FrM && taranto's lat/lon in 99_clean.xlsx
    fig.savefig(kwargs["output_file"], dpi=600)


def update_left_with_merged_info(left, right):
    debug(f"{sorted(left.columns, key=str.lower)=}")
    debug(f"{sorted(right.columns, key=str.lower)=}")
    assert "color" in set(right.columns) & set(left.columns)

    debug(f"left_df  valid predicates={left.sindex.valid_query_predicates}")
    debug(f"right_df valid predicates={right.sindex.valid_query_predicates}")
    merged = left.sjoin(
        right,
        how="inner",  # intersecates keys from both dfs + left_df geometry col
        predicate="contains",
    ).sort_index()
    debug(
        f"right_df\n{right.to_string()}\n\nmerged_df=\n"
        + merged.loc[
            :,
            ["color_left", "color_right"],
        ].to_string()
        + f"\n\n{merged.shape=}\n"
    )
    duplicated_list = merged[
        merged.duplicated(subset=["OBJECTID"], keep=False) == True
    ]
    debug(f"languages in same position\n{duplicated_list.to_string()}")
    for i, (left_idx, right_color) in enumerate(
        merged.loc[merged["color_right"].notna(), ["color_right"]].itertuples()
    ):
        # I don't completely understand why this is necessary, but still...
        if left.loc[left_idx, "color"] != right_color:
            debug(
                f"{i:03d})\tleft_df.loc[{left_idx=}, 'color'] "
                f"\t<~\t{right_color=}"  # fmt: skip
            )
            left.loc[left_idx, "color"] = right_color
    return left.copy(deep=True)


if __name__ == "__main__":
    parser = get_cli_parser(__doc__, __file__)
    parser.add_argument(
        "-g",
        "--geo-coordinates",
        default=__DEFAULT["geo_coordinates"],
        help="Excel file containing latitude/longitude information",
        metavar="xlsx",
        type=FileType("rb"),
    )
    parser.add_argument(
        "-i",
        "--input-file",
        default=__DEFAULT["input_file"],
        help="Clusters produced by 01_plot_clusters.py",
        metavar="txt",
        required=True,
        type=FileType("r"),
    )
    parser.add_argument(
        "-m",
        "--map-file",
        default=__DEFAULT["map_file"],
        help="Shapefile used to gelocalize clusters",
        metavar="shp",
        type=FileType("rb"),
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default=__DEFAULT["output_file"],
        help="Plot of geolocalized clusters (allowed extensions: "
        + ", ".join(sorted(__DEFAULT["allowed_extensions"]))
        + ")",
        metavar="path",
        required=True,
        type=FileType("wb"),
    )
    parser.add_argument(
        "-k",
        "--do-markers",
        default=__DEFAULT["do_markers"],
        help="Plot markers on the map",
        metavar="True/False",
        type=boolean_value,
    )
    parser.add_argument(
        "-e",
        "--do-areas",
        default=__DEFAULT["do_areas"],
        help="Plot areas on the map",
        metavar="True/False",
        type=boolean_value,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=__DEFAULT["verbose"],
    )

    parsed_args = parser.parse_args()  # parse CLI arguments

    initialize_logging(
        basename(__file__).removesuffix(".py").rstrip("_") + "__debug.log",
        parsed_args.verbose,
    )
    debug(f"{parsed_args=}")

    thematic_map(**vars(parsed_args))
