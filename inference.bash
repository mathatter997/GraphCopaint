# /bin/bash

# ddim
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 1000 \
#  --sampler ddim \
#  --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small_ddim.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --cpu False --num_samples 1000 \
#  --sampler ddim \
#  --checkpoint_path models/Ego_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_ddim.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type enzyme  --cpu False --num_samples 113 \
#  --sampler ddim \
#  --checkpoint_path models/ENZYMES/gnn/checkpoint_epoch_10000_t1000_psgn.pth \
#  --scheduler_path models/ENZYMES/scheduler_config.json \
#  --output_path data/dataset/output_enyzme_ddim.json

# # ddpm 
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 1000 \
#  --sampler ddpm \
#  --checkpoint_path models/Community_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small_ddpm.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --cpu False --num_samples 1000 \
#  --sampler ddpm \
#  --checkpoint_path models/Ego_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_ddpm.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type enzyme  --cpu False --num_samples 113 \
#  --sampler ddpm \
#  --checkpoint_path models/ENZYMES/gnn/checkpoint_epoch_10000_t1000_psgn.pth \
#  --scheduler_path models/ENZYMES/scheduler_config.json \
#  --output_path data/dataset/output_enyzme_ddpm.json

#  CUDA_VISIBLE_DEVICES="2" python inference.py --config_type ego  --cpu False --num_samples 1000 \
#  --checkpoint_path models/Ego/gnn/checkpoint_epoch_900_t1000_psgn.pth \
#  --scheduler_path models/Ego/scheduler_config.json \
#  --output_path data/dataset/output_ego.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 10 \
#  --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
#  --loss_mode naive_inpaint --reg_mode naive_square \
#  --num_intervals 1 --optimization_steps 3 --tau 5 --time_travel True \
#  --checkpoint_path models/Community_small/gnn/checkpoint_epoch_300000_t1000_psgn.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small_copaint_s1.json \
#  --mask_path data/dataset/mask_com_small_copaint_s1.json \
#  --masked_output_path data/dataset/masked_com_small_copaint_s1.json \
#  --log_x0_predictions True

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu False --num_samples 1000 \
#  --sampler ddim --use_copaint True --num_timesteps 1000 \
#  --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel False \
#  --checkpoint_path models/Community_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small_copaint.json \
#  --mask_path data/dataset/mask_com_small_copaint.json \
#  --masked_output_path data/dataset/masked_com_small_copaint.json 

# python inference.py --config_type community_small  --cpu True --num_samples 1 \
#  --sampler ddim --num_timesteps 1000 --log_x0_predictions True \
#  --checkpoint_path models/Community_small/gnn/checkpoint_epoch_300000_t1000_psgn.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small_s1.json

# python inference.py --config_type community_small  --cpu True --num_samples 1000 \
#  --sampler ddpm --num_timesteps 25 \
#  --checkpoint_path models/Community_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small_t25.json

# python inference.py --config_type ego_small  --cpu True --num_samples 1000 \
#  --sampler ddim --num_timesteps 25 \
#  --checkpoint_path models/Ego_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_ddim_t25.json

# python inference.py --config_type ego  --cpu True --num_samples 1000 \
#  --sampler ddim --num_timesteps 25 \
#  --checkpoint_path models/Ego/gnn/checkpoint_epoch_1000_t1000_psgn.pth \
#  --scheduler_path models/Ego/scheduler_config.json \
#  --output_path data/dataset/output_ego_ddim_t25.json


# loss_mode="naive_inpaint"
# reg_mode="naive_square"
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 32 \
#     --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
#     --loss_mode $loss_mode --reg_mode $reg_mode \
#     --lr_xt_decay 1.0 --use_adaptive_lr_xt True \
#     --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
#     --checkpoint_path models/Community_small/gnn/checkpoint_epoch_300000_t1000_psgn.pth \
#     --scheduler_path models/Community_small/scheduler_config.json \
#     --output_path data/dataset/ablation/output_com_small_copaint_a6_0.json \
#     --mask_path data/dataset/ablation/mask_com_small_copaint_a6_0.json \
#     --masked_output_path data/dataset/ablation/masked_com_small_copaint_a6_0.json \
#     --log_x0_predictions False

# loss_mode="naive_inpaint"
# reg_mode="naive_square"
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 1 \
#     --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
#     --loss_mode $loss_mode --reg_mode $reg_mode \
#     --lr_xt_decay 1.0 --use_adaptive_lr_xt True \
#     --num_intervals 10 --optimization_steps 0 --tau 1 --time_travel False \
#     --checkpoint_path models/Community_small/gnn/checkpoint_epoch_300000_t1000_psgn.pth \
#     --scheduler_path models/Community_small/scheduler_config.json \
#     --output_path data/dataset/ablation/output_com_small_copaint_a6_0.json \
#     --mask_path data/dataset/ablation/mask_com_small_copaint_a6_0.json \
#     --masked_output_path data/dataset/ablation/masked_com_small_copaint_a6_0.json \
#     --log_x0_predictions True

# loss_mode="naive_inpaint"
# reg_mode="naive_square"
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small_smooth  --cpu True --num_samples 1 \
#     --sampler ddim --inpainter 'none' --num_timesteps 1000 \
#     --loss_mode $loss_mode --reg_mode $reg_mode \
#     --lr_xt_decay 1.0 --use_adaptive_lr_xt True \
#     --num_intervals 10 --optimization_steps 0 --tau 1 --time_travel False \
#     --checkpoint_path models/Community_small/gnn/checkpoint_epoch_300000_t1000_psgn.pth \
#     --scheduler_path models/Community_small/scheduler_config.json \
#     --output_path data/dataset/ablation/output_com_small_copaint_a6_0.json \
#     --mask_path data/dataset/ablation/mask_com_small_copaint_a6_0.json \
#     --masked_output_path data/dataset/ablation/masked_com_small_copaint_a6_0.json \
#     --log_x0_predictions True

# --checkpoint_path models/Community_small_smooth/gnn/checkpoint_epoch_20000_t1000_psgn.pth \
# --scheduler_path models/Community_small_smooth/scheduler_config.json \
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 1 \
#     --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
#     --loss_mode $loss_mode --reg_mode $reg_mode \
#     --lr_xt_decay 1.0 --use_adaptive_lr_xt True \
#     --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
#     --checkpoint_path models/Community_small/gnn/checkpoint_epoch_300000_t1000_psgn.pth \
#     --scheduler_path models/Community_small_smooth/scheduler_config.json \
#     --output_path data/dataset/output_com_small_copaint_test.json \
#     --mask_path data/dataset/mask_com_small_copaint_test.json \
#     --masked_output_path data/dataset/masked_com_small_copaint_test.json \
#     --log_x0_predictions False


# loss_mode="naive_inpaint"
# reg_mode="naive_square"
# data_name="mnist_zeros"
# inpainter="none"
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type mnist_zeros  --cpu False --num_samples 1 \
#     --sampler ddim --inpainter ${inpainter} --num_timesteps 1000 \
#     --loss_mode $loss_mode --reg_mode $reg_mode \
#     --lr_xt_decay 1.0 --use_adaptive_lr_xt True \
#     --num_intervals 10 --optimization_steps 0 --tau 1 --time_travel False \
#     --checkpoint_path models/${data_name}/gnn/checkpoint_epoch_20000_t1000_psgn.pth \
#     --scheduler_path models/${data_name}/scheduler_config.json \
#     --output_path data/dataset/ablation/output_${data_name}_${inpainter}_a1_0.json \
#     --mask_path data/dataset/ablation/mask_${data_name}_${inpainter}_a1_0.json \
#     --masked_output_path data/dataset/ablation/masked_${data_name}_${inpainter}_a1_0.json \
#     --log_x0_predictions True

# loss_mode="naive_inpaint"
# reg_mode="naive_square"
# data_name="mnist_zeros"
# inpainter="none"
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ${data_name}  --cpu True --num_samples 1 \
#     --sampler ddim --inpainter ${inpainter} --num_timesteps 1000 \
#     --loss_mode $loss_mode --reg_mode $reg_mode \
#     --lr_xt_decay 1.0 --use_adaptive_lr_xt True \
#     --num_intervals 10 --optimization_steps 0 --tau 1 --time_travel False \
#     --checkpoint_path models/${data_name}/gnn/checkpoint_epoch_300000_t1000_psgn.pth \
#     --scheduler_path models/${data_name}/scheduler_config.json \
#     --output_path data/dataset/ablation/output_${data_name}_${inpainter}_a1_0.json \
#     --mask_path data/dataset/ablation/mask_${data_name}_${inpainter}_a1_0.json \
#     --masked_output_path data/dataset/ablation/masked_${data_name}_${inpainter}_a1_0.json \
#     --log_x0_predictions True

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --cpu False --num_samples 1 \
#  --sampler ddim --log_x0_predictions True \
#  --checkpoint_path models/Ego_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_ddim_s1.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --cpu True --num_samples 1 \
#  --sampler ddpm --log_x0_predictions True --max_n_nodes 18 \
#  --checkpoint_path models/Ego_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_ddim_s1.json


# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --use_ema True --num_samples 1000 \
#  --sampler vpsde --log_x0_predictions False \
#  --checkpoint_path models/Community_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small_eigen_vpsde_bm5_ep_100000_clamp.json

#  CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --use_ema True --num_samples 1000 \
#  --sampler ddpm --log_x0_predictions False \
#  --checkpoint_path models/Community_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small_eigen_ddpm_bm5_ep_100000_clamp.json

#  CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --use_ema True --num_samples 1000 \
#  --sampler ddim --log_x0_predictions False \
#  --checkpoint_path models/Community_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small_eigen_ddim_bm5_ep_100000_clamp.json


# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --use_ema True --cpu True --num_samples 1000 \
#  --sampler vpsde --log_x0_predictions False --max_n_nodes 18 \
#  --checkpoint_path models/Ego_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_vpsde.json

#  SIBLE_DEVICES="0" python inference.py --config_type ego_small  --use_ema True --cpu True --num_samples 1000 \
#  --sampler ddpm --log_x0_predictions False --max_n_nodes 18 \
#  --checkpoint_path models/Ego_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_ddpm.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --use_ema True --cpu True --num_samples 1000 \
#  --sampler ddim --log_x0_predictions False --max_n_nodes 18 \
#  --checkpoint_path models/Ego_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_ddim.json


# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --cpu True --num_samples 1 \
#  --sampler ddim --log_x0_predictions True \
#  --checkpoint_path models/Ego_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_ddim_s1.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 1 \
#  --sampler ddim --log_x0_predictions True --max_n_nodes 20 --use_ema True\
#  --alpha 0.1\
#  --checkpoint_path models/Community_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/output_com_small_ddim_eigen_special_1.json


# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --cpu True --num_samples 1 \
#  --sampler ddim --log_x0_predictions True --max_n_nodes 18 \
#  --checkpoint_path models/Ego_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/output_ego_small_vpsde_s1.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --cpu True --num_samples 1 \
#  --sampler ddim --log_x0_predictions True --max_n_nodes 18 \
#  --checkpoint_path models/Ego_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/output_ego_small_ddim_special.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 1 \
#  --sampler ddim --log_x0_predictions True --max_n_nodes 20 --alpha 0.5\
#  --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/output_com_small_ddim_special_1.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type enzyme  --cpu True --num_samples 113 \
#  --sampler ddpm --log_x0_predictions False --max_n_nodes 125 \
#  --checkpoint_path models/ENZYMES/gnn/diag_vpsde_bm20_5000.pth \
#  --scheduler_path models/ENZYMES/scheduler_config.json \
#  --output_path data/output_enzyme_vpsde_bm20_s113.json


# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type grid  --cpu True --num_samples 1 \
#  --sampler ddpm --log_x0_predictions True --max_n_nodes 361 \
#  --checkpoint_path models/grid/gnn/diag_vpsde_bm20_5000.pth \
#  --scheduler_path models/grid/scheduler_config.json \
#  --output_path data/output_grid_vpsde_bm20_s1.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --use_ema True --cpu True --num_samples 1000 \
#  --sampler ddpm --log_x0_predictions True --max_n_nodes 18 \
#  --checkpoint_path models/Ego_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_ddpm.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --use_ema True --cpu True --num_samples 1000 \
#  --sampler ddim --log_x0_predictions True --max_n_nodes 18 \
#  --checkpoint_path models/Ego_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_ddim.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --use_ema True --cpu True --num_samples 1000 \
#  --sampler vpsde --log_x0_predictions True --max_n_nodes 18 \
#  --checkpoint_path models/Ego_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small_vpsde.json

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --use_ema True --cpu True --num_samples 100 \
#  --sampler ddpm --log_x0_predictions True --max_n_nodes 20 \
#  --checkpoint_path models/Community_small/gnn/diag_vpsde_bm5_100000.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small_ddpm.json

# data_name='grid'
# inpainter='copaint'
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ${data_name}  --cpu True --num_samples 1 \
#  --sampler ddpm --inpainter 'copaint' --log_x0_predictions True --max_n_nodes 361 \
#  --checkpoint_path models/grid/gnn/diag_vpsde_bm5_10000.pth \
#  --scheduler_path models/grid/scheduler_config.json \
#  --output_path data/output_grid_vpsde_bm5_s1.json\
#  --output_path data/dataset/ablation/output_${data_name}_${inpainter}_a1_0.json \
#  --mask_path data/dataset/ablation/mask_${data_name}_${inpainter}_a1_0.json \
#  --masked_output_path data/dataset/ablation/masked_${data_name}_${inpainter}_a1_0.json

CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 1 \
    --sampler ddim --alpha 0.5 --inpainter 'none' --num_timesteps 1000 \
    --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
    --scheduler_path models/Community_small/scheduler_config.json \
    --output_path data/output_com_small.json\