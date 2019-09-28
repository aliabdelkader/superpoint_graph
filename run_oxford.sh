export OXFORD_DIR="./"

python supervized_partition/graph_processing.py --ROOT_PATH $OXFORD_DIR --dataset oxford --voxel_width 0.03 --use_voronoi 1 --plane_model 0