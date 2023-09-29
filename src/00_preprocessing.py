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
   Preprocess datasets/01..04*.xlsx to have research-ready datasets

   Usage:
            coming soon...
"""

from argparse import FileType
from logging import debug, info, warning  # , critical
from os.path import basename
from string import ascii_letters, punctuation
from utility import (
    ALLOWED_EXTENSIONS,
    get_cli_parser,
    initialize_logging,
    MISSING_VALUES,
    serialize,
    jaccard,
)
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

parser = get_cli_parser(__doc__, __file__)

parser.add_argument(
    "-i",
    "--input",
    default=None,
    help="Excel input file containing the DataFrame to parse",
    metavar="xlsx",
    required=True,
    type=FileType("rb"),
)
parser.add_argument(
    "-s",
    "--sheet",
    default=0,
    help="Sheet number or name to read from unput file",
    metavar="int|str",
)
parser.add_argument(
    "-p",
    "--pivot-cell",
    default="TP",
    dest="pivot",
    help="Content of the most upper-left cell in the table",
    type=str,
)
parser.add_argument(
    "--pivot-cell-type",
    choices=("record", "feature", "row", "col"),
    default="record",
    dest="pivot_axis",
    help="If 'row' or 'record' the matrix will be transposed",
)
parser.add_argument(
    "-f",
    "--flatten-multi-index",
    default=["NUM Pellegrini", "Isogloss", "Code", "Label"],
    dest="flat_index",
    help="Preferred order of columns guiding the multi-index substitution",
    metavar="col",
    nargs="+",
    type=str,
)
grp = parser.add_mutually_exclusive_group(required=True)
grp.add_argument(
    "-0",
    "--drop-nan",
    action="store_true",
    help="Drop columns/features with missing/null/NaN values",
)
grp.add_argument(
    "-k",
    "--impute-nan",
    default=0,
    dest="n_neighbors",
    help="Impute missing data with K nearest neighbors",
    metavar="int",
    type=int,
)
parser.add_argument(
    "-o",
    "--output",
    help="Output file (allowed extensions: "
    + ", ".join(sorted(ALLOWED_EXTENSIONS.keys()))
    + ";\nuse - to serialize in pickle format to <stdout>)",
    metavar="file",
    required=True,
    type=FileType("wb"),
)
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    dest="verbosity",
)
parsed_args = parser.parse_args()  # parse CLI arguments

initialize_logging(
    basename(__file__).removesuffix(".py").rstrip("_") + "__debug.log",
    parsed_args.verbosity,
)
debug(f"{parsed_args=}")

# Discover available sheets
sheets = set(pd.read_excel(parsed_args.input, sheet_name=None).keys())
debug(f"Available sheets: {repr(sorted(sheets, key=str.lower))[1:-1]}.")
assert parsed_args.sheet in sheets | set(
    range(len(sheets))
), f"Required --sheet={parsed_args.sheet!r} not found!"

excel_kwargs = dict(
    sheet_name=parsed_args.sheet,
    true_values=["+", "Yes", "yes"],
    false_values=["-", "No", "no"],
    na_values=MISSING_VALUES,
)
df = pd.read_excel(parsed_args.input, **excel_kwargs)
debug(
    "Hypothetical columns:\n\t"
    + "\n\t".join(
        [
            f"{row_i=}\t"
            + ", ".join(
                map(
                    lambda s: f"{str(s)[:9]}{'...' if len(str(s))>9 else ''}",
                    df.iloc[row_i, :].astype("string").to_list(),
                )
            )
            for row_i in range(min(5, len(df.index)))
        ]
    )
)
debug(
    "Hypothetical index:\n"
    + pd.concat(
        [
            pd.DataFrame(np.vstack([df.columns, df]))
            .iloc[:, col_j]
            .rename(f"{col_j=}")
            .astype("string")
            .str.slice_replace(32, repl="...")
            for col_j in range(min(5, len(df.columns)))
        ],
        axis=1,
    ).to_string()
)

# Discover the actual start of the structured table
expected_columns = None
excel_kwargs["index_col"] = 0
excel_kwargs["skiprows"] = 0
if df.reset_index(drop=True).duplicated(keep=False).any():
    excel_kwargs["skipfooter"] = [
        i for i, flag in df.reset_index(drop=True).duplicated().items() if flag
    ]
try:
    for col_j in range(len(df.columns)):
        for row_i in range(len(df.index)):
            if str(df.iloc[row_i, col_j]) != parsed_args.pivot:
                continue
            debug(f"{row_i=}, {col_j=}")
            excel_kwargs["index_col"] = list(range(col_j))
            excel_kwargs["skiprows"] = row_i + 1
            expected_columns = df.iloc[
                row_i, list(range(col_j, len(df.columns)))
            ].to_list()
            raise StopIteration(
                "Pivot matched cell "
                + repr(":".join(map(str, (chr(ord("A") + col_j), row_i + 1))))
            )
except StopIteration as e:
    info(str(e))  # empty rows are skipped, thus the row may be smaller!
    if "skipfooter" in excel_kwargs:
        excel_kwargs["skipfooter"] = set(excel_kwargs.pop("skipfooter")) - set(
            range(excel_kwargs["skiprows"])
        )
        if not excel_kwargs["skipfooter"]:
            excel_kwargs.pop("skipfooter")  # empty set
        elif excel_kwargs["skipfooter"] == set(
            range(min(excel_kwargs["skipfooter"]), len(df.index))
        ):
            excel_kwargs["skipfooter"] = len(df.index) - min(
                excel_kwargs["skipfooter"],
            )
        else:
            warning(
                "Could not selectively skip non-adjacent rows "
                + repr(tuple(excel_kwargs.pop("skipfooter")))
            )
    debug(excel_kwargs)
else:
    warning(
        f"Could not find pivot {parsed_args.pivot!r} in sheet "
        f"{parsed_args.sheet!r} of {parsed_args.input.name!r}"
    )
    raise SystemExit(1)

# Final painless (hopefully) parsing of the table
df = pd.read_excel(parsed_args.input, **excel_kwargs)
debug(
    "Dataset contains the following values: "
    + repr(sorted(set(df.values.flatten()), key=str))[1:-1]
    + "."
)
if expected_columns is not None and set(df.columns) - set(expected_columns):
    warning(
        "Could not find the following expected columns:\n\t-"
        + "\n\t-".join(
            sorted(
                set(df.columns) - set(expected_columns),
                key=str.lower,
            )
        )
    )

# Avoid using multi-index, let's arbitrarily choose the left most one
if len(df.index.names) > 1:
    warning(f"Rows have multiple indexes: {repr(list(df.index.names))[1:-1]}!")
    for col in parsed_args.flat_index:
        if col in df.index.names:
            df = df.reset_index(
                list(set(df.index.names) - set([col])),
                drop=True,
            )
            break
    else:
        df = df.reset_index(list(df.index.names)[:-1], drop=True)
    info(f"Column {df.index.name!r} will be used instead")

# Drop duplicated headers (again)
drop_rows = [
    i
    for i, row in df.reset_index(drop=True).transpose().items()
    if df.shape[1] == row.size and row.eq(df.columns).all()
]
if drop_rows:
    warning(
        "Dropping rows duplicating the header:\n"
        + df.reset_index().iloc[drop_rows, :].to_string()
    )
    df = df.iloc[sorted(set(range(df.shape[0])) - set(drop_rows)), :]

# sort index (rows) and columns
df = (
    df.loc[:, sorted(df.columns, key=str.lower)]
    .reset_index(names="index")
    .assign(
        sort_by=lambda _df: pd.Series(
            [
                str(
                    int(
                        _df.loc[i, "index"]
                        .split("=")[0]
                        .strip(ascii_letters + punctuation)
                    )
                ).rjust(2, "0")
                if "=" in str(_df.loc[i, "index"])
                else str("0" if flag else "")
                for i, flag in _df["index"].astype(str).str.len().eq(1).items()
            ],
            dtype="string",
        ).str.cat(_df["index"].astype("string"), join="right")
    )
    .sort_values("sort_by")
    .drop(columns="sort_by")  # comment this line to debug index sorting
    .set_index("index")
    .rename_axis(df.index.name, axis=0)
)
# Somehow input 03 and 04 fail automagically converting some of the
# true/false values; maybe because of the duplicated header, which
# however we already dropped at this point
df = df.replace({tv: True for tv in excel_kwargs["true_values"]}).replace(
    {fv: False for fv in excel_kwargs["false_values"]}
)

if parsed_args.pivot_axis in ("record", "row"):
    debug(f"Transposing dataset because of {parsed_args.pivot_axis=}")
    df = df.transpose()

if parsed_args.drop_nan:
    debug("Dropping columns/features with missing data")
    df = df.dropna(axis=1)
elif parsed_args.n_neighbors > 0:
    df = df.rename(columns=str)

    # keep_empty_features=True => if a feature is full NaN, it is set to False
    # metric can also be set to "nan_euclidean"
    imputer = KNNImputer(
        n_neighbors=parsed_args.n_neighbors,
        weights="distance",
        metric=jaccard,
        keep_empty_features=True,
    )
    imputed_array = imputer.fit_transform(df)

    df = pd.DataFrame(imputed_array, columns=df.columns, index=df.index).ge(
        0.5
    )

else:
    parser.error("The number of neighbors must be > 0")

debug(f"final dataframe\n{df.to_string()}\n\n{df.shape=}")

if set(df.values.flatten()) != {True, False}:
    warning(
        "Dataset contains the following non-boolean values: "
        + repr(sorted(set(df.values.flatten()) - {True, False}, key=str))[1:-1]
        + "."
    )

df = df.rename(index={"PCM" : "PCa", "TE" : "Ter"})

try:
    serialize(df, parsed_args.output)
except Exception as e:
    parser.error(str(e))
