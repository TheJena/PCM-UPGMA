#!/bin/bash

SCRIPT_DIR="src"
DATASET_DIR="datasets/last_email"
PREPROCESSED_DIR="out_preprocess"
PLOT_DIR="out_plot"

INPUT_FILE_LIST=$(cd ${DATASET_DIR} && ls -v1 *.xlsx | fgrep -v "~")
PYTHON_FILE_MAP=02_carta_diocesi.py

rm -rf ${PREPROCESSED_DIR} ${PLOT_DIR}
mkdir ${PREPROCESSED_DIR}
mkdir ${PLOT_DIR}

# Preprocess all dataset with knn-imputer
for file in ${INPUT_FILE_LIST}; do
	python3 ${SCRIPT_DIR}/preprocessing.py		\
		-i ${DATASET_DIR}"/"${file}		\
		-o ${PREPROCESSED_DIR}"/"${file}	\
		-k 5					\
		-vv					\
	|| exit 1;
done

python3 ${SCRIPT_DIR}/01_plot_clusters.py	\
	-i ${PREPROCESSED_DIR}			\
	-o ${PLOT_DIR}				\
	-p					\
	-vv					\
|| exit 2

python3 ${SCRIPT_DIR}/${PYTHON_FILE_MAP}	\
	-i ${PLOT_DIR}/clusters.txt		\
	-o ${PLOT_DIR}/mappa_clusters.pdf	\
	-vv					\
|| exit 3

for i in `seq 0 3`; do
	python3 ${SCRIPT_DIR}/${PYTHON_FILE_MAP}	\
		-i ${PLOT_DIR}/clusters_${i}.txt	\
		-o ${PLOT_DIR}/mappa_clusters_${i}.pdf	\
		-v					\
	|| exit 4;
done
