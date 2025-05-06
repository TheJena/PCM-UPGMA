#!/bin/bash

SCRIPT_DIR="src"
DATASET_DIR="datasets"
PREPROCESSED_DIR="out_preprocess"
PLOT_DIR="out_plot"

INPUT_FILE_LIST=$(cd ${DATASET_DIR} && ls -1 {01,02,03,04}_new*.xlsx)
PYTHON_FILE_MAP=02_carta_diocesi.py
#PYTHON_FILE_MAP=02_carta_italia.py

rm -rf ${PREPROCESSED_DIR} ${PLOT_DIR}
mkdir ${PREPROCESSED_DIR}
mkdir ${PLOT_DIR}

# Preprocess all dataset with knn-imputer
for file in ${INPUT_FILE_LIST}; do
	python3 ${SCRIPT_DIR}/preprocessing.py		\
		-i ${DATASET_DIR}"/"${file}		\
		-o ${PREPROCESSED_DIR}"/"${file}	\
		-k 5;
done

python3 ${SCRIPT_DIR}/01_plot_clusters.py	\
	-i ${PREPROCESSED_DIR}			\
	-o ${PLOT_DIR}				\
	-p

python3 ${SCRIPT_DIR}/${PYTHON_FILE_MAP}	\
	-i ${PLOT_DIR}/clusters.txt		\
	-o ${PLOT_DIR}

# for i in `seq 1 4`; do
# 	python3 ${SCRIPT_DIR}/${PYTHON_FILE_MAP}	\
# 		-i ${PLOT_DIR}/clusters_${i}.txt	\
# 		-o ${PLOT_DIR}
# done
