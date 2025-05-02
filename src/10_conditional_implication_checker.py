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
from logging import debug, warning
from os.path import basename
from utility import get_cli_parser, initialize_logging
import pandas as pd

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
    default="params",
    help="Sheet number or name to read from --input-matrix",
    metavar="str",
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

# Discover available sheets
df_dict = pd.read_excel(parsed_args.input_rules, sheet_name=None)

df = df_dict.get(parsed_args.rules_sheet, None)
if df is None:
    debug(
        "Available sheets: "
        + repr(
            sorted(
                set(df_dict.keys()),
                key=str.lower,
            )
        )[1:-1]
    )
    parser.error(
        f"Could not find sheet {parsed_args.rules_sheet!r}. "
        "Please increase verbosity to see available sheets (above)"
    )
del df_dict


for col in (parsed_args.parameter, parsed_args.formula):
    assert col in df.columns, str(
        f"Column {col!r} not found in {parsed_args.input_rules.name}"
    )

impl_dict = (
    df.set_index(
        parsed_args.parameter,
        drop=True,
    )
    .loc[:, parsed_args.formula]
    .sort_index()
    .to_dict()
)
for k, v in impl_dict.items():
    debug(f"{k:<8}\t{v!r}")
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

# Discover available sheets
df_dict = pd.read_excel(parsed_args.input_matrix, sheet_name=None)

df = df_dict.get(parsed_args.matrix_sheet, None)
if df is None:
    debug(
        "Available sheets: "
        + repr(
            sorted(
                set(df_dict.keys()),
                key=str.lower,
            )
        )[1:-1]
    )
    parser.error(
        f"Could not find sheet {parsed_args.matrix_sheet!r}. "
        "Please increase verbosity to see available sheets (above)"
    )

param_col = df.columns[0]

df = (
    df.assign(**{param_col: lambda _df: _df[param_col].str.upper()})
    .sort_values(param_col)
    .set_index(param_col)
)
debug(f"\n{df.to_string()}")


for lang in sorted(df.columns, key=str.lower):
    lang_dict = df[lang].to_dict()
    # debug(f"[{lang}]\t\t{lang_dict!r}".replace("'", ""))
    for param, formula in impl_dict.items():
        param_value_in_lang = lang_dict[param]
        original_formula = str(formula)
        for k, v in lang_dict.items():
            formula = formula.replace(k, repr(v))
        formula_result = (formula == "") or eval(formula)

        if not formula_result and param_value_in_lang in ("0",):
            continue  # implication verified
        elif formula_result and param_value_in_lang in ("+", "-"):
            continue  # implication verified
        warning(
            f"Conditional implication failed in language {lang!r}:\n\t"
            f"{param:<4} := {original_formula}\n\t"
            f"{param_value_in_lang:^3} =!= {formula}\n\t"
        )
