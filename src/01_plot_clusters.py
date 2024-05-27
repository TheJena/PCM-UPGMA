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
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from utility import get_cli_parser
import os

# ---- MACRO ----

DPI         = 300
PLT_TITLE   = "Hierarchical Clustering Dendrogram\n(UPGMA method)"
CHOSEN_DISTANCE = "hamming"

# -- END MACRO --


# ---- START ARGDEF ----

parser = get_cli_parser(__doc__, __file__)

parser.add_argument(
    "-i",
    "--input_directory",
    default="./out_preprocess",
    help="Directory containing preprocessed (with 00_preprocess) files",
    metavar="str",
    required=True,
    type=str,
)

parser.add_argument(
    "-o",
    "--output_directory",
    default="./out_plot",
    help="Directory that will contain results ()",
    metavar="str",
    required=True,
    type=str,
)

parser.add_argument(
    "-p",
    "--print_clusterless",
    default=False,
    help="Print clusterless dialects",
    action="store_true",
)

parsed_args = parser.parse_args()

# ----- END ARGDEF -----


# ----- START SCRIPT -----

def plot_dendrogram(
    model,                
    plot_title,
    plot_x_label,
    out_file_name,
    color_threshold_multiplier,
    **kwargs
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
    plt.xlabel(plot_x_label)
    
    if color_threshold_multiplier is None:
        color_threshold = None
    else:
        color_threshold = color_threshold_multiplier * max(linkage_matrix[:,2])
    out_dendogram = dendrogram(linkage_matrix, color_threshold=color_threshold, **kwargs)

    plt.savefig(out_file_name, dpi=DPI)

    return out_dendogram


def read_and_make_dendrogram(
    input_file_name,
    out_file_name,
    plot_title,
    plot_x_label,
    input_sheet_name="Sheet1",
):
    df = pd.read_excel(input_file_name, sheet_name=input_sheet_name)
    df = df.set_index(df.columns[0]).astype(int)
    df = df.sort_index()
    df.columns = df.columns.astype(str)

    model = AgglomerativeClustering(
        linkage="average",
        metric=CHOSEN_DISTANCE,
        distance_threshold=0,
        n_clusters=None,
    )

    model.fit(df)

    languages = list(df.index)
    dendrogram = plot_dendrogram(
        model,
        plot_title,
        plot_x_label,
        out_file_name,
        0.65,
        labels=languages
    )

    distances = dict()
    for language in languages:
        distances[language] = distance.cdist(
            df.values,
            np.array([df.loc[language]]),
            metric=CHOSEN_DISTANCE,
        )

    return [distances, dendrogram]

def are_same_cluster(source_dialect, dest_dialect, groupings, k, no_groups_index):
    try:
        source_color = groupings[k][source_dialect]
        dest_color = groupings[k][dest_dialect]
        return (source_color != no_groups_index) and (source_color == dest_color)
    except:
        pass
    return False

def create_clusters(clusters_dict, all_languages, valid_languages): 
    clusterless_list = list()
    result_last = dict()
    for language in all_languages:
        if language not in valid_languages:
            continue
        sorted_list = sorted(set(clusters_dict[language]))
        if len(sorted_list) > 1:
            el = sorted_list[0]
            result_last[sorted_list[0]] = sorted_list
        else:
            clusterless_list += [language]
    return clusterless_list, result_last

def get_output_clusters(clusters_dict, parsed_args, all_languages, valid_languages):
    clusterless_list, result_last = create_clusters(clusters_dict, all_languages, valid_languages)
    out_string = ""
    for key in result_last.keys():
        for el in result_last[key]:
            out_string += (el + " ")
        out_string += "\n"

    if parsed_args.print_clusterless and len(clusterless_list) > 0:
        out_string += "C:\n"
        for el in clusterless_list:
            out_string += el + " "
        out_string += "\n"
    return out_string

input_file_list = os.listdir(parsed_args.input_directory)
num_plots = len(input_file_list)

results     = list()
groupings   = list()
for i in range(num_plots):
    results += [
        read_and_make_dendrogram(
            parsed_args.input_directory + "/" + input_file_list[i],
            parsed_args.output_directory + "/" + 
                input_file_list[i].replace(".xlsx", ".png"),
            PLT_TITLE,
            input_file_list[i].rstrip(".xlsx").replace("_", " "),
        )
    ]

    groupings += [dict()]
    for j in range(len(results[i][1]["ivl"])):
        color = results[i][1]["leaves_color_list"][j]
        dialect = results[i][1]["ivl"][j]
        groupings[i][dialect] = color
        
total_num_matches = dict()

reported = set()
languages = list(groupings[0].keys())
for i in range(1, num_plots):
    languages_own = list()
    for l in groupings[i].keys():
        if l not in languages and l not in reported:
            print(l + " not in all tables!")
            languages += [l]
            reported.add(l)
    for l in languages:
        if l not in groupings[i].keys() and l not in reported:
            print(l + " not in all tables!")
            reported.add(l)
languages = sorted(languages)
shared_languages = []
for l in languages:
    if l not in reported:
        shared_languages += [l]
shared_languages = sorted(shared_languages)
clusters = [
    {l : [] for l in languages}
    for _ in range(num_plots + 1)
]

no_groups_index = "C0"
for source_dialect in languages:
    num_matches = []
    for dest_dialect in languages:
        num_single_matches = 0
        for k in range(num_plots):
            if are_same_cluster(source_dialect, dest_dialect, groupings, k, no_groups_index):
                num_single_matches += 1
        num_matches += [num_single_matches]
        clusters[num_single_matches][source_dialect] += [dest_dialect]
    total_num_matches[source_dialect] = num_matches

final_result = pd.DataFrame.from_dict(
    total_num_matches, orient="index", columns=languages
)

final_result.to_excel(
    parsed_args.output_directory + "/plot_clusters_result.xlsx"
)

out_string = get_output_clusters(clusters[num_plots], parsed_args, languages, shared_languages)

with open(parsed_args.output_directory + "/clusters.txt", "w") as f:
    f.write(str(num_plots) + " Tables Fused\n" + out_string)

for k in range(num_plots):
    clusters = {l : [] for l in languages}
    for source_dialect in languages:
        for dest_dialect in languages:
            if are_same_cluster(source_dialect, dest_dialect, groupings, k, no_groups_index):
                clusters[source_dialect] += [dest_dialect]

    out_string = get_output_clusters(clusters, parsed_args, languages, groupings[k].keys())

    with open(parsed_args.output_directory + "/clusters_" + str(k) + ".txt", "w") as f:
        f.write(input_file_list[k].rstrip(".xlsx").replace("_", " ") + "\n" + out_string)
