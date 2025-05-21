#!/usr/bin/env bash

if test "$#" -ne 1 ; then
    echo -e "Usage:\n\t${0} INPUT_DIR" >&2
    exit 1
fi
if ! test -d "${1}"; then
    echo -e "Error:\tINPUT_DIR=${1} not found" >&2
    exit 2
fi

DATASET_DIR="${1}"
INPUT_FILE_LIST=$(cd "${DATASET_DIR}" && ls -v1 0[1234]_*.xlsx | fgrep -v "~")
PLOT_SCRIPT="${PLOT_SCRIPT:-src/02_carta_diocesi.py}"

echo "Using PLOT_SCRIPT=${PLOT_SCRIPT}" >&2

PLOT_DIR="out_plot"
PREPROCESSED_DIR="out_preprocess"
PYTHON_FILE_MAP="${PLOT_SCRIPT}"
SCRIPT_DIR="src"

rm -rf   ${PREPROCESSED_DIR} ${PLOT_DIR}
mkdir -p ${PREPROCESSED_DIR} ${PLOT_DIR}

# Preprocess all dataset with knn-imputer
for file in ${INPUT_FILE_LIST}; do
    python3 ${SCRIPT_DIR}/preprocessing.py        \
        -i ${DATASET_DIR}"/"${file}               \
        -o ${PREPROCESSED_DIR}"/"${file}          \
        -k 5                                      \
        -v                                        \
    || exit 1;
done

python3 ${SCRIPT_DIR}/01_plot_clusters.py         \
    -i ${PREPROCESSED_DIR}                        \
    -o ${PLOT_DIR}                                \
    -p                                            \
    -v                                            \
|| exit 2

python3 ${PYTHON_FILE_MAP}                        \
    -i ${PLOT_DIR}/clusters.txt                   \
    -o ${PLOT_DIR}/mappa_clusters.pdf             \
    -v                                            \
|| exit 3

for i in `seq 0 3`; do
    python3 ${PYTHON_FILE_MAP}                    \
        -i ${PLOT_DIR}/clusters_${i}.txt          \
        -o ${PLOT_DIR}/mappa_clusters_${i}.pdf    \
        -v                                        \
    || exit 4;
done
