export CUDA_VISIBLE_DEVICES=0,1,2
export OXFORD_DIR="./"

# python supervized_partition/graph_processing.py --ROOT_PATH $OXFORD_DIR --dataset oxford --voxel_width 0 --use_voronoi 1 --plane_model 0

python ./supervized_partition/supervized_partition.py --ROOT_PATH $OXFORD_DIR --dataset oxford \
    --epochs 100 --test_nth_epoch 10 --reg_strength 0.5 --spatial_emb 0.02 --batch_size 10 \
    --global_feat exyrgb --CP_cutoff 10 --odir results_part/oxford/best/ --cuda 1
