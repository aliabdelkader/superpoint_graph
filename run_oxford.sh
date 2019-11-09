
export OXFORD_DIR="oxford"

# bash ./run_prepare_oxford

# python supervized_partition/graph_processing.py --ROOT_PATH $OXFORD_DIR --dataset oxford --voxel_width 0 --use_voronoi 1 --plane_model 0

export CUDA_VISIBLE_DEVICES=0



# python ./supervized_partition/supervized_partition.py --ROOT_PATH $OXFORD_DIR --dataset oxford \
#     --epochs 100 --test_nth_epoch 1 --reg_strength 0.5 --spatial_emb 0.02 --batch_size 24 \
#     --global_feat exyrgb --CP_cutoff 10 --odir results_part/oxford/best/ --cuda 1 --nworkers 2

# python ./supervized_partition/generate_partition.py --modeldir results_part/oxford/best/ --cuda 1 --input_folder $OXFORD_DIR/features_supervision \
# --overwrite 1 --nworkers 2

# python ./learning/oxford_dataset.py --OXFORD_PATH $OXFORD_DIR


# python ./learning/main.py --dataset oxford --OXFORD_PATH $OXFORD_DIR --epochs 100 \ --lr_steps "[40, 50, 60, 70, 80]" --test_nth_epoch 1 --model_config gru_10_1_1_1_0,f_19 --pc_attribs xyzXYZrgb \ 
# --ptn_nfeat_stn 9 --batch_size 10 --ptn_minpts 15 --spg_augm_order 3 --spg_augm_hardcutoff 256 \
# --ptn_widths "[[64,64,128], [64,32,32]]" --ptn_widths_stn "[[32,64], [32,16]]" --loss_weights sqrt \
# --use_val_set 1 --odir results/oxford/best/; \


# python partition/partition.py --dataset oxford --ROOT_PATH $OXFORD_DIR --voxel_width 0.01 --reg_strength 0.8 --ver_batch 0

# python learning/oxford_dataset.py --OXFORD_PATH $OXFORD_DIR

# python learning/main.py --dataset oxford --OXFORD_PATH $OXFORD_DIR --db_test_name testset --db_train_name trainset \
# --epochs 500 --lr_steps '[350, 400, 450]' --test_nth_epoch 1 --model_config 'gru_10,f_19' --ptn_nfeat_stn 11 \
# --nworkers 2 --pc_attrib xyzrgbelpsv --odir "results/oxford/trainval_best" --use_val_set 1


python learning/main.py --dataset oxford --OXFORD_PATH $OXFORD_DIR --db_test_name testset --db_train_name trainset \
--epochs -1 --lr_steps '[350, 400, 450]' --test_nth_epoch 1 --model_config 'gru_10,f_19' --ptn_nfeat_stn 11 \
--nworkers 1 --pc_attrib xyzrgbelpsv --odir "results/oxford/trainval_best" --resume RESUME --cuda 1 --test_multisamp_n 1 \