#!/usr/bin/env python3
# coding: utf-8
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2023 Federico Motta    <federico.motta@unimore.it>
#                    Lorenzo  Carletti <lorenzo.carletti@unimore.it>
#                    Matteo   Vanzini  <matteo.vanzini@unimore.it>
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
            coming soon...
"""
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    """Create linkage matrix and then plot the dendrogram"""

    # create the counts of samples under each node
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
    
    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)

def read_and_make_dendrogram(filename, sheet, do_transpose, problem_char, xlabel, title, out_filename, dpi=1200):
    dfs = pd.read_excel(filename, sheet_name=None)
    df = dfs[sheet]

    df_t = df
    if do_transpose:
        df_t = df.transpose()

    # Row/Column index removal
    if do_transpose:
        df_t.columns = df_t.iloc[0, :].to_list()
        df_t = df_t.iloc[list(range(1, 33)), :]
    else:
        df_t.index = df_t.iloc[: , 0].to_list()
        df_t = df_t.iloc[: , list(range(1, len(df_t.keys())))]
    
    df_t = df_t.sort_index()

    selector = [param for param, flag in df_t.astype(str).eq(problem_char).sum(axis=0).eq(0).to_dict().items() if flag]
    df_bool_imputed = df_t.loc[:, selector].replace({"+": True, "-": False, "1": True, "0": False, 1: True, 0: False}).astype("boolean")

    df_clust = df_bool_imputed.astype(int)
    model = AgglomerativeClustering(linkage="average", metric="euclidean", distance_threshold=0, n_clusters=None)
    model.fit(df_clust)
    plt.clf()
    plt.cla()
    plt.title(title)
    languages = list(df_clust.transpose().columns)
    resulting_dendrogram = plot_dendrogram(model, labels=languages)
    plt.xlabel(xlabel)
    plt.savefig(out_filename, dpi=dpi)
    resulting_distances = dict()
    # There is probably a faster/smarter/better way to do this and/or using the dendrogram itself
    for s_language in languages:
        resulting_distances[s_language] = distance.cdist(df_clust.values, np.array([df_clust.loc[s_language]]), metric='euclidean')
    return [resulting_distances, resulting_dendrogram]

base_folder = "datasets/"
base_file = ["preprocessed_inputs_01.xlsx", "preprocessed_inputs_02.xlsx", "preprocessed_inputs_03.xlsx", "preprocessed_inputs_04.xlsx"]
sheet_name = ["Foglio1", "Foglio1", "Foglio1", "no_all_zero_rules_imputed_PGE_W"]
need_to_transpose = [False, False, False, True]
unallowed_char = ["?", "?", "?", "0"]
output_names = ["TMP", "TMP", "TMP", "32 dialetti, from 'TableA 2023', imputed (PGE=-, WAP=++-, GFL=-, PGL=+)"]
titles = ["Hierarchical Clustering Dendrogram\n(UPGMA method)", "Hierarchical Clustering Dendrogram\n(UPGMA method)", "Hierarchical Clustering Dendrogram\n(UPGMA method)", "Hierarchical Clustering Dendrogram\n(UPGMA method)"]

num_plots = 4
results = []
groupings = []
no_groups_index = "C0"

for i in range(num_plots):
    results += [read_and_make_dendrogram(base_folder + base_file[i], sheet_name[i], need_to_transpose[i], unallowed_char[i], output_names[i], titles[i], "./plot" + str(i + 1) + ".svg")]
    groupings += [[dict(), dict()]]
    for j in range(len(results[i][1]['ivl'])):
        color = results[i][1]['leaves_color_list'][j]
        dialect = results[i][1]['ivl'][j]
        groupings[i][0][dialect] = color
        if(color not in groupings[i][1].keys()):
            groupings[i][1][color] = []
        groupings[i][1][color] += [dialect]

total_num_matches = dict()

# This is a dumb way to do it, but it's good enough to start this out.
# A more appropriate way may use results[k][0] as weights,
# which I computed for this specific reason
languages = list(groupings[0][0].keys())
languages.sort()
clusters = []
for i in range(num_plots + 1):    
    cluster = dict()
    for _ in languages:
        cluster[_] = []
    clusters += [cluster]

for source_dialect in languages:
    num_matches = []
    for dest_dialect in languages:
        num_single_matches = 0
        for k in range(num_plots):
            source_color = groupings[k][0][source_dialect]
            dest_color = groupings[k][0][dest_dialect]
            if (source_color != no_groups_index) and (source_color == dest_color):
                num_single_matches += 1
        num_matches += [num_single_matches]
        clusters[num_single_matches][source_dialect] += [dest_dialect]
    total_num_matches[source_dialect] = num_matches

final_result = pd.DataFrame.from_dict(total_num_matches, orient="index", columns=languages)
final_result.to_excel("result.xlsx")

for _ in languages:
    print(_ + ": " + str(clusters[num_plots][_]))
