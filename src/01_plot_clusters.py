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
Build a dendrogram by running UPGMA over some language parameters

UPGMA (Unweighted Pair Group Method with Arithmetic mean) is a
simple agglomerative (bottom-up) hierarchical clustering method

Usage:
    export PREPROCESSED_DIR="out_preprocess"; \
    export PLOT_DIR="out_plot";               \
    mkdir -p "${PLOT_DIR}";                   \
    && python3 src/01_plot_clusters.py        \
        -i "${PREPROCESSED_DIR}"              \
        -o "${PLOT_DIR}"                      \
        --print-clusterless
"""
from collections import Counter, defaultdict
from functools import cache
from itertools import chain
from logging import debug, info  # , warning, critical
from matplotlib import colormaps
from matplotlib import colors
from matplotlib import pyplot as plt
from os import listdir
from os.path import basename, join as join_path
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from utility import get_cli_parser, initialize_logging
import numpy as np
import pandas as pd


__DEFAULT = dict(
    chosen_distance="hamming",
    dpi=300,
    input_directory="./out_preprocess",
    output_directory="./out_plot",
    plot_title="Hierarchical Clustering Dendrogram\n(UPGMA method)",
    print_clusterless=False,
    verbose=0,
)


def get_all_shared_languages(dendrograms):
    count_languages = Counter(
        chain.from_iterable(d.keys() for d in lang2grp_mapping(dendrograms))
    )
    debug(f"{count_languages=}")

    shared_langs, reported_langs = set(), set()
    for lang, qty in count_languages.most_common():
        if qty == len(lang2grp_mapping(dendrograms)):
            shared_langs.add(lang)
        else:
            reported_langs.add(lang)
            info(
                f"{lang:<8} only found in "
                f"{qty:>3d}/{len(lang2grp_mapping(dendrograms)):<3d} tables!"
            )
    debug(f"{reported_langs=}")

    all_langs = sorted(set(shared_langs) | set(reported_langs), key=str.lower)
    shared_langs = sorted(shared_langs, key=str.lower)
    debug(f"\n{all_langs=}\n{shared_langs=}")
    return all_langs, shared_langs


def get_dendrogram_and_distances(
    input_file_name,
    out_file_name,
    x_label,
    sheet_name=0,
    **kwargs,
):
    df = pd.read_excel(input_file_name, sheet_name=sheet_name)
    debug(f"df({input_file_name})\n{df.to_string()}")

    df = (  # set languages as index
        df.set_index(df.columns[0]).sort_index().rename_axis(index="languages")
    )
    languages = df.index.to_series().to_list()

    df = df.astype(int)  # cast booleans to inteers
    df = df.rename(columns=lambda i: str(i))  # int2str column renaming
    debug(f"df=\n{df.to_string()}")

    model = AgglomerativeClustering(
        linkage="average",
        metric=kwargs["chosen_distance"],
        distance_threshold=0,
        n_clusters=None,
    ).fit(df)
    debug(f"{model.get_params()=}")

    dendrogram = plot_dendrogram(
        model,
        kwargs["plot_title"],
        x_label,
        out_file_name,
        dpi=kwargs["dpi"],
        labels=languages,
        # color_threshold=0.7,
        # link_color_func=link_color_func,
    )
    debug(f"{sorted(dendrogram.keys())=}")

    distances = {
        lang: distance.cdist(
            df.to_numpy(),
            df.loc[[lang]].to_numpy(),
            metric=kwargs["chosen_distance"],
        )
        for lang in languages
    }
    debug(
        "distances=\n"
        + "\n".join(
            f"{lang}: [\n {str(matrix.T)[1:-1]}\n].T"
            for lang, matrix in distances.items()
        )
    )
    return dendrogram, distances


def lang2grp_mapping(dendrograms, use_cache=True):
    global __CACHED_DENDROGRAMS
    if not use_cache or globals().get("__CACHED_DENDROGRAMS", None) is None:
        __CACHED_DENDROGRAMS = [
            defaultdict(
                lambda: "MISSING LANGUAGE->COLOR MAPPING",
                **dict(zip(d["ivl"], d["leaves_color_list"])),
            )
            for d in dendrograms
        ]
        debug(
            f"dendrograms lang2grp mapping={__CACHED_DENDROGRAMS}".replace(
                "default", "\n\n\tdefault"
            )
        )
    return __CACHED_DENDROGRAMS


def link_color_func(k):
    return colors.to_hex(colormaps.get_cmap("jet")(hash(k)))


def plot_clusters(**kwargs):
    for k, v in kwargs.items():
        assert k in __DEFAULT, str(
            "Unrecognized parameter"
            f" {plot_clusters.__name__}( ... {k}={v!r} ... )"
            f"\nPlease use only the available ones:\n\t"
            + "\n\t".join(sorted(__DEFAULT.keys(), key=str.lower))
        )
    for k, v in __DEFAULT.items():
        if k not in kwargs:
            kwargs[k] = v
    debug(f"\n\n\nCalling {plot_clusters.__name__}(")
    for k, v in kwargs.items():
        debug(f"{k:<16} = {v!r}")
    debug(")\n\n\n")

    info(f"Reading excel files from {kwargs['input_directory']!s}")
    input_file_list = [
        f
        for f in sorted(listdir(kwargs["input_directory"]), key=str.lower)
        if not f.startswith("~$")
    ]
    info("\n\t" + "\n\t".join(input_file_list))
    num_plots = len(input_file_list)

    dendrograms, distances = split_zipped_tuples(
        get_dendrogram_and_distances(
            join_path(kwargs["input_directory"], input_file),
            join_path(
                kwargs["output_directory"],
                input_file.replace(".xlsx", ".svg"),
            ),
            input_file.removesuffix(".xlsx").replace("_", " "),
            **kwargs,
        )
        for input_file in input_file_list
    )
    del distances  # apparently unused (?)

    clusterless_color = "C0"
    same_cluster_co_occurrences = dict()
    clusters = [defaultdict(list) for _ in range(num_plots + 1)]
    all_langs, shared_langs = get_all_shared_languages(dendrograms)
    for src_lang in all_langs:
        num_matches = list()
        for dst_lang in all_langs:
            num_single_matches = sum(
                same_color(grp[src_lang], grp[dst_lang], clusterless_color)
                for grp in lang2grp_mapping(dendrograms)
            )
            num_matches.append(num_single_matches)
            clusters[num_single_matches][src_lang].append(dst_lang)
        same_cluster_co_occurrences[src_lang] = num_matches
    debug(f"{clusters=}".replace("default", "\n\n\tdefault"))

    co_occurrences_df = pd.DataFrame.from_dict(
        same_cluster_co_occurrences, orient="index", columns=all_langs
    )
    debug(f"co_occurrences_df=\n{co_occurrences_df.to_string()}")
    co_occurrences_df.to_excel(
        join_path(
            kwargs["output_directory"],
            "languages_co_occurrences_across_clusters.xlsx",
        )
    )

    serialize_clusters(
        clusters[num_plots],
        shared_langs,
        path=join_path(kwargs["output_directory"], "clusters.txt"),
        header=f"{num_plots!s} Tables Fused",
        print_clusterless=kwargs["print_clusterless"],
    )

    for k, (filename, grp) in enumerate(
        zip(input_file_list, lang2grp_mapping(dendrograms))
    ):
        serialize_clusters(
            {
                src_lang: [
                    dst_lang
                    for dst_lang in all_langs
                    if same_color(
                        grp[src_lang],
                        grp[dst_lang],
                        clusterless_color,
                    )
                ]
                for src_lang in all_langs
            },
            grp.keys(),
            path=join_path(
                kwargs["output_directory"],
                f"clusters_{k+1}.txt",
            ),
            header=filename.removesuffix(".xlsx").replace("_", " "),
            print_clusterless=kwargs["print_clusterless"],
        )


def plot_dendrogram(
    model,
    plot_title,
    x_label,
    output_file,
    color_threshold_coeff=0.65,
    dpi=300,
    **kwargs,
):
    """Create linkage matrix and then plot the dendrogram"""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    plt.clf()
    plt.cla()
    plt.title(plot_title)
    plt.xlabel(x_label)

    color_threshold = None
    if color_threshold_coeff is not None:
        color_threshold = color_threshold_coeff * max(linkage_matrix[:, 2])

    debug(f"{kwargs!r}")
    out_dendogram = dendrogram(
        linkage_matrix, color_threshold=color_threshold, **kwargs
    )
    plt.savefig(output_file, dpi=dpi)
    return out_dendogram


@cache  # just removes redundant debugging messages
def same_color(src_color, dst_color, clusterless_color=None):
    ret = (
        (clusterless_color is None) or (src_color != clusterless_color)
    ) and (  # fmt: skip
        src_color == dst_color
    )
    debug(f"{src_color=}\t{dst_color=}\t{clusterless_color=}\t{ret=}")
    return ret


def serialize_clusters(
    clusters_dict,
    lang_subset,
    path,
    header,
    print_clusterless=False,
):
    clusters, clusterless_list = dict(), list()
    for lang in sorted(set(lang_subset), key=str.lower):
        sorted_list = sorted(set(clusters_dict[lang]), key=str.lower)
        if not sorted_list or sorted_list == [lang]:
            clusterless_list += [lang]
        else:
            clusters[sorted_list[0]] = sorted_list

    debug(f"{header=}")
    body = "\n".join(" ".join(langs) for langs in clusters.values())
    if print_clusterless and len(clusterless_list):
        body += f"\nC:\n{' '.join(clusterless_list)}"
    debug(f"body=\n{body.strip()}\n{'#' * 80}")

    with open(path, "w") as f:
        f.write(f"{header.strip()}\n{body.strip()}")


def split_zipped_tuples(generator):
    return map(list, zip(*list(generator)))


if __name__ == "__main__":
    parser = get_cli_parser(__doc__, __file__)

    parser.add_argument(
        "-i",
        "--input-directory",
        default=__DEFAULT["input_directory"],
        help="Directory containing preprocessed files",
        metavar="path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        default=__DEFAULT["output_directory"],
        help="Directory that will contain results",
        metavar="path",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-p",
        "--print-clusterless",
        default=__DEFAULT["print_clusterless"],
        help="Print clusterless dialects",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",  # dest="verbosity",
        action="count",
        default=__DEFAULT["verbose"],
    )
    parsed_args = parser.parse_args()

    initialize_logging(
        basename(__file__).removesuffix(".py").rstrip("_") + "__debug.log",
        parsed_args.verbose,
    )
    debug(f"{parsed_args=}")

    plot_clusters(**vars(parsed_args))
