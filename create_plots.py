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
    dendrogram(linkage_matrix, **kwargs)

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
    plot_dendrogram(model, labels=list(df_clust.transpose().columns))
    plt.xlabel(xlabel)
    plt.savefig(out_filename, dpi=dpi)

base_folder = "datasets/"
base_file = ["preprocessed_inputs_01.xlsx", "preprocessed_inputs_02.xlsx", "preprocessed_inputs_03.xlsx", "preprocessed_inputs_04.xlsx"]
sheet_name = ["Foglio1", "Foglio1", "Foglio1", "no_all_zero_rules_imputed_PGE_W"]
need_to_transpose = [False, False, False, True]
unallowed_char = ["?", "?", "?", "0"]
output_names = ["TMP", "TMP", "TMP", "32 dialetti, from 'TableA 2023', imputed (PGE=-, WAP=++-, GFL=-, PGL=+)"]
titles = ["Hierarchical Clustering Dendrogram\n(UPGMA method)", "Hierarchical Clustering Dendrogram\n(UPGMA method)", "Hierarchical Clustering Dendrogram\n(UPGMA method)", "Hierarchical Clustering Dendrogram\n(UPGMA method)"]

num_plots = 4

for i in range(num_plots):
    read_and_make_dendrogram(base_folder + base_file[i], sheet_name[i], need_to_transpose[i], unallowed_char[i], output_names[i], titles[i], "./plot" + str(i + 1) + ".svg")
