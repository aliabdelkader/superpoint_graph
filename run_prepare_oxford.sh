
export BASE_DIR="/home/fusionresearch/Rafaat/oxford-annotation/oxford-automation/results"
export CHUNK_NUM="2014-05-14-13-59-05"
export INCLUDE_INDEX="123"
export DATASET_DIR="${BASE_DIR}/${CHUNK_NUM}"
export OUTPUT_DIR="data/${CHUNK_NUM}"

echo "${DATASET_DIR}" 

python prepare_oxford.py  --dataset_files "${DATASET_DIR}" --output_dir "${OUTPUT_DIR}" --include "${INCLUDE_INDEX}"