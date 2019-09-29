
export BASE_DIR="/home/fusionresearch/Rafaat/oxford-annotation/oxford-automation/results"

# chunk 1
echo "chunk 1"
export CHUNK_NUM="2014-06-24-14-20-41"
export INCLUDE_INDEX="100"
export DATASET_DIR="${BASE_DIR}/${CHUNK_NUM}"
export OUTPUT_DIR="data/${CHUNK_NUM}"

python prepare_oxford.py  --dataset_files "${DATASET_DIR}" --output_dir "${OUTPUT_DIR}" --include "${INCLUDE_INDEX}"

# chunk 2

echo "chunk 2"
export CHUNK_NUM="2014-05-14-13-59-05"
export INCLUDE_INDEX="123"
export DATASET_DIR="${BASE_DIR}/${CHUNK_NUM}"
export OUTPUT_DIR="data/${CHUNK_NUM}"

python prepare_oxford.py  --dataset_files "${DATASET_DIR}" --output_dir "${OUTPUT_DIR}" --include "${INCLUDE_INDEX}"


# chunk 3

echo "chunk 3"
export CHUNK_NUM="2014-05-06-13-14-58"
export INCLUDE_INDEX="100"
export DATASET_DIR="${BASE_DIR}/${CHUNK_NUM}"
export OUTPUT_DIR="data/${CHUNK_NUM}"

python prepare_oxford.py  --dataset_files "${DATASET_DIR}" --output_dir "${OUTPUT_DIR}" --include "${INCLUDE_INDEX}"


# chunk 4

# echo "chunk 4"
# export CHUNK_NUM="2014-06-25-16-22-15"
# export INCLUDE_INDEX="100"
# export DATASET_DIR="${BASE_DIR}/${CHUNK_NUM}"
# export OUTPUT_DIR="data/${CHUNK_NUM}"

# python prepare_oxford.py  --dataset_files "${DATASET_DIR}" --output_dir "${OUTPUT_DIR}" --include "${INCLUDE_INDEX}"