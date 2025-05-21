clear && for f in datasets/*.gpg; do git log --follow --format=%ad --date iso $f | tail -n1 | tr -d "\n"; echo -e "\t\t${f/.gpg/}"; done | fgrep -iv preprocessed | fgrep -v 99 | sort -V
# 2023-06-29 13:21:21 +0200	datasets/01_Guardiano_et_al_Pellegrini_1970.xlsx
# 2023-06-29 13:21:21 +0200	datasets/02_Guardiano_et_al_Pellegrini_1977.xlsx
# 2023-06-29 13:21:21 +0200	datasets/03_Guardiano_et_al_SSWL_DPProperties.xlsx
# 2023-06-29 13:21:21 +0200	datasets/04_Guardiano_et_al_Table_A_2023_04_03.xlsx
# 2023-06-29 13:21:21 +0200	datasets/Table_A_transactions.xlsx
#
# 2023-06-30 18:49:28 +0200	datasets/99_km_distances_among_dialects.xlsx
#
# 2024-03-19 22:55:54 +0100	datasets/01_new_Pellegrini_1970_MATRIX.xlsx
# 2024-03-19 22:55:54 +0100	datasets/02_new_Pellegrini_1977_Matrix.xlsx
# 2024-03-19 22:55:54 +0100	datasets/03_new_SSWL_DP_Table.xlsx
# 2024-03-19 22:55:54 +0100	datasets/04_new_TableA.xlsx
#
# 2024-03-19 22:55:54 +0100	datasets/99_dialects_lat_long_geolocation_clean.xlsx
#
# 2025-04-11 20:20:00 +0100	datasets/last_email/Pellegrini_1970.xlsx
# 2025-04-11 20:20:00 +0100	datasets/last_email/Pellegrini_1977.xlsx
# 2025-04-11 20:20:00 +0100	datasets/last_email/SSWL.xlsx
# 2025-04-11 20:20:00 +0100	datasets/last_email/TableA_2025SI.xlsx

export BANNER=$(yes "#" | tr -d "\n" | head -c 80)
export CHECKER="src/10_conditional_implication_checker.py"

export LAST_TABLE_A="datasets/last_email/TableA_2025SI.xlsx"
export PENULTIMATE_TABLE_A="datasets/04_new_TableA.xlsx"
export FIRST_TABLE_A="datasets/04_Guardiano_et_al_Table_A_2023_04_03.xlsx"

echo -e "${BANNER}\n${BANNER}\n${BANNER}\n${BANNER}"
python3 "${CHECKER}"			\
	-i "${LAST_TABLE_A}"		\
	-s "TABLE A_2024 (2)"		\
	-c "Implicational Condition(s)"	\
	-I "${LAST_TABLE_A}"		\
	-v				\
2>&1 | tee checked_last.log
echo "Press Ctrl+D to continue"; cat

echo -e "${BANNER}\n${BANNER}\n${BANNER}\n${BANNER}"
python3 "${CHECKER}"			\
	-i "${FIRST_TABLE_A}"		\
	-s "TABLE A_2023"		\
	-I "${PENULTIMATE_TABLE_A}"	\
	-S "Italia+GRK"			\
	-fv				\
2>&1 | tee checked_penultimate.log
echo "Press Ctrl+D to continue"; cat

echo -e "${BANNER}\n${BANNER}\n${BANNER}\n${BANNER}"
python3 "${CHECKER}"			\
	-i "${FIRST_TABLE_A}"		\
	-s "TABLE A_2023"		\
	-I "${FIRST_TABLE_A}"		\
	-S "TABLE A_2023"		\
	-fv				\
2>&1 | tee checked_first.log
