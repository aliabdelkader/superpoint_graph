export CUDA_VISIBLE_DEVICES=0
export OXFORD_DIR="oxford"

# python supervized_partition/graph_processing.py --ROOT_PATH $OXFORD_DIR --dataset oxford --voxel_width 0 --use_voronoi 1 --plane_model 0

export CUDA_VISIBLE_DEVICES=0

python ./supervized_partition/supervized_partition.py --ROOT_PATH $OXFORD_DIR --dataset oxford \
    --epochs 100 --test_nth_epoch 1 --reg_strength 0.5 --spatial_emb 0.02 --batch_size 24 \
    --global_feat exyrgb --CP_cutoff 10 --odir results_part/oxford/best/ --cuda 1 --nworkers 2

python ./supervized_partition/generate_partition.py --modeldir results_part/oxford/best/ --cuda 1 --input_folder $OXFORD_DIR/features_supervision \
--overwrite 1 --nworkers 2

python ./learning/oxford_dataset.py --OXFORD_PATH $OXFORD_DIR
OXFORD_PATH

python ./learning/main.py --dataset oxford --OXFORD_PATH $OXFORD_DIR --epochs 100 \ --lr_steps "[40, 50, 60, 70, 80]" --test_nth_epoch 1 --model_config gru_10_1_1_1_0,f_19 --pc_attribs xyzXYZrgb \ 
--ptn_nfeat_stn 9 --batch_size 10 --ptn_minpts 15 --spg_augm_order 3 --spg_augm_hardcutoff 256 \
--ptn_widths "[[64,64,128], [64,32,32]]" --ptn_widths_stn "[[32,64], [32,16]]" --loss_weights sqrt \
--use_val_set 1 --odir results/oxford/best/; \