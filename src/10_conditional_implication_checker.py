#!/usr/bin/env python3
# coding: utf-8
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2025 Federico Motta    <federico.motta@unimore.it>
#                    Lorenzo  Carletti <lorenzo.carletti@unimore.it>
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
Usage:
         coming soon...
"""

from argparse import FileType
from logging import debug, info, warning
from os.path import basename
from utility import get_cli_parser, initialize_logging
import pandas as pd

from preprocessing import clean_excel_file

parser = get_cli_parser(__doc__, __file__)

parser.add_argument(
    "-i",
    "--input-rules",
    default=None,
    help="Excel input file with the conditional implication table to parse",
    metavar="xlsx",
    required=True,
    type=FileType("rb"),
)
parser.add_argument(
    "-s",
    "--rules-sheet",
    default="rules",
    help="Sheet name to read from --input-rules file",
    metavar="str",
)
parser.add_argument(
    "-p",
    "--implied-parameter",
    default="Label",
    dest="parameter",
    help="Implied parameter column name in --rules-sheet",
    metavar="str",
    type=str,
)
parser.add_argument(
    "-c",
    "--implication-condition",
    default="Implicational condition(s)",
    dest="formula",
    help="Implication formula column name in --rules-sheet",
    metavar="str",
    type=str,
)
parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    help="Force usage of implicational condition(s) index "
    "(Input matrix index will be ignored)",
)
parser.add_argument(
    "-I",
    "--input-matrix",
    default=None,
    help="Excel input file containing the language matrix to parse",
    metavar="xlsx",
    required=True,
    type=FileType("rb"),
)
parser.add_argument(
    "-S",
    "--matrix-sheet",
    default="TABLE A_2024 (2)",
    help="Sheet number or name to read from --input-matrix",
    metavar="int|str",
)
parser.add_argument(
    "-P",
    "--matrix-pivot",
    default="TP",
    help="First parameter column name in --matrix-sheet",
    metavar="str",
    type=str,
)
parser.add_argument(
    "-m",
    "--minus-symbols",
    default=["-", "−"],
    help="Equivalent symbols all meaning the '-PARAM' expression",
    metavar="glyph",
    nargs="+",
    type=str,
)
parser.add_argument(
    "-n",
    "--not-symbol",
    default="¬",
    help="Symbol meaning the '¬PARAM' expression",
    metavar="glyph",
    type=str,
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

df = clean_excel_file(
    input=parsed_args.input_rules,
    sheet=parsed_args.rules_sheet,
    pivot_cell="Label",
    pivot_cell_type="feature",
    maintain_original_values=True,
    sort_rows=False,
    verbose=0,
)

for col in (parsed_args.parameter, parsed_args.formula):
    assert col in df.columns, str(
        f"Column {col!r} not found in {parsed_args.input_rules.name}"
    )

parameter_default_order = df.loc[:, parsed_args.parameter].dropna()
debug(f"parameter_default_order={parameter_default_order.to_list()!r}")

impl_dict = (
    df.set_index(
        parsed_args.parameter,
        drop=True,
    )
    .loc[:, parsed_args.formula]
    .sort_index()
    .to_dict()
)
for k in sorted(impl_dict.keys(), key=lambda v: repr(v).lower()):
    v = impl_dict.pop(k)
    if k not in set(parameter_default_order) or pd.isna(k):
        debug(f"Dropping useless Label={k!r} ~> Formula={v!r}")
        continue
    debug(f"{k:<8}\t{v!r}")
    assert isinstance(k, str), repr(k)
    impl_dict[k] = v
debug("\n\n")
del df


for param in sorted(impl_dict.keys(), key=str.lower):
    formula = impl_dict.pop(param)
    param = param.upper()
    if pd.isna(formula):
        formula = ""
    for glyph in parsed_args.minus_symbols:
        formula = formula.replace(glyph, "-")
    formula = (
        formula.replace(parsed_args.not_symbol, "not ")
        .replace(",", " and ")
        .replace("(", " ( ")
        .replace(")", " ) ")
    )

    for token in formula.split():
        if token in ("(", ")", "not", "or", "and"):
            pass
        elif token.startswith("+"):
            formula = formula.replace(
                token,
                f"({token.lstrip('+')}=='+')",
            )
        elif token.startswith("-"):
            formula = formula.replace(
                token,
                f"({token.lstrip('-')}=='-')",
            )
        else:
            warning(f"Unrecognized token {token!r}")
    formula = " ".join(formula.split())
    debug(f"{param:^3} = {formula!r}")

    # TODO: implement a lexical parser robust to potentially
    # ambiguous, missing parenthesis, e.g. through
    # https://docs.python.org/3/library/shlex.html

    impl_dict[param] = formula


df = clean_excel_file(
    input=parsed_args.input_matrix,
    sheet=parsed_args.matrix_sheet,
    pivot_cell=parsed_args.matrix_pivot,
    pivot_cell_type="feature",
    maintain_original_values=True,
    sort_rows=False,
    sort_columns=True,
    verbose=0,
)

df = df.dropna(axis="columns", thresh=min(5, df.shape[0]))
debug(f"df=\t#Dropped cols with <= 5 non-missing values\n{df.to_string()}")

for i in range(df.shape[0]):
    if df.iloc[i, :].eq(list(df.columns)).sum() >= 0.5:
        info(f"Dropping redundant row:\n\t{df.iloc[i, :].to_list()!s}")
        df = df.iloc[[j for j in range(df.shape[0]) if i != j], :]

for col in sorted(df.columns):
    if df[col].eq(df.index.values).sum() >= 0.5:
        info(f"Dropping redundant column {col!r}")
        df = df.drop(columns=col)

if not (
    df.index.size == parameter_default_order.size
    and all(
        a == b
        for a, b in zip(
            df.index.to_series().to_list(), parameter_default_order.to_list()
        )
    )
):
    warning(
        f"The index of the --input-matrix (size={df.index.size}) "
        f"is not exactly equal to the one found in --input-rules "
        f"(size={parameter_default_order.size})!\n"
        f"Use (-v/--verbose) twice to see a value-by-value comparison"
    )
    debug(
        "\n"
        + pd.concat(
            [
                pd.Series(df.index.values),
                pd.Series(parameter_default_order.to_list()),
            ],
            axis=1,
            ignore_index=True,
        )
        .rename(columns=dict(enumerate(["input_matrix", "input_rules"])))
        .assign(
            DIFFERING=lambda _df: (
                _df.input_matrix != _df.input_rules
            ).replace({True: "YES", False: ""})
        )
        .to_string()
    )
    info(
        "Use -f/--force to override the index found in input matrix "
        "with the one found in input rules"
    )
    if not parsed_args.force:
        raise SystemExit(11)

if set(df.index.values) & set(parameter_default_order.to_list()):
    # leveraging only the overlapping values
    df = df.loc[
        [i for i in parameter_default_order.to_list() if i in df.index.values],
        :,
    ]
else:  # truncating (since hopefully longer) input matrix
    assert len(df.index.values) > len(parameter_default_order.to_list()), str(
        "Conditional implications are about more parameters than "
        "those found in input matrix!\nPlease provide either less "
        "conditional implications or more parameters within input "
        "matrix"
    )
    df = df.iloc[range(parameter_default_order.size), :].set_index(
        pd.Index(parameter_default_order.to_list())
    )
parameter_default_order = list(df.index.values)
debug(f"{parameter_default_order=}")

df = (
    df.reset_index(drop=False, names="Parameter")
    .assign(
        **{
            "Parameter": lambda _df: _df["Parameter"]
            .astype("string")
            .str.upper()
        }
    )
    .sort_values("Parameter")
    .set_index("Parameter")
)
debug(f"\n{df.to_string()}")

failed_implications = 0
for lang_name, lang_params in df.items():
    lang_params = lang_params.to_dict()
    debug(f"[{lang_name}]\t=\t{lang_params!r}".replace("'", ""))
    for param, formula in impl_dict.items():
        if param not in lang_params:
            warning(
                f"Skipping missing parameter {param:<4} "
                f"in language {lang_name!r}"
            )
            continue
        param_value_in_lang = str(lang_params[param])
        original_formula = str(formula)
        for k, v in lang_params.items():
            formula = formula.replace(k, repr(v))
        formula_result = (formula == "") or eval(formula)

        if not formula_result and param_value_in_lang in ("0",):
            continue  # implication verified
        elif formula_result and param_value_in_lang in ("+", "-"):
            continue  # implication verified
        warning(
            f"Conditional implication failed in language {lang_name!r}:\n\t"
            f"{param:<4} := {original_formula}\n\t"
            f"{param_value_in_lang:^3} =!= {formula}\n\t"
        )
        failed_implications += 1

if failed_implications == 0:
    info(
        "All the conditional implications were verified "
        "in the provided languages :)"  # fmt: skip
    )
