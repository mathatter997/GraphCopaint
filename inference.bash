# /bin/bash

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 1000 \
#  --checkpoint_path models/Community_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Community_small/scheduler_config.json \
#  --output_path data/dataset/output_com_small.json

#  CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --cpu False --num_samples 1000 \
#  --checkpoint_path models/Ego_small/gnn/checkpoint_epoch_200000_t1000_psgn.pth \
#  --scheduler_path models/Ego_small/scheduler_config.json \
#  --output_path data/dataset/output_ego_small.json

#  CUDA_VISIBLE_DEVICES="2" python inference.py --config_type ego  --cpu False --num_samples 1000 \
#  --checkpoint_path models/Ego/gnn/checkpoint_epoch_900_t1000_psgn.pth \
#  --scheduler_path models/Ego/scheduler_config.json \
#  --output_path data/dataset/output_ego.json
 
#  CUDA_VISIBLE_DEVICES="0" python inference.py --config_type enzyme  --cpu False --num_samples 1000 \
#  --checkpoint_path models/ENZYMES/gnn/checkpoint_epoch_10000_t1000_psgn.pth \
#  --scheduler_path models/ENZYMES/scheduler_config.json \
#  --output_path data/dataset/output_enyzme.json

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

loss_mode="naive_inpaint"
reg_mode="naive_square"
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
CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small_smooth  --cpu True --num_samples 1 \
    --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
    --loss_mode $loss_mode --reg_mode $reg_mode \
    --lr_xt_decay 1.0 --use_adaptive_lr_xt True \
    --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
    --checkpoint_path models/Community_small_smooth/gnn/checkpoint_epoch_20000_t1000_psgn.pth \
    --scheduler_path models/Community_small_smooth/scheduler_config.json \
    --output_path data/dataset/ablation/output_com_small_copaint_a6_0.json \
    --mask_path data/dataset/ablation/mask_com_small_copaint_a6_0.json \
    --masked_output_path data/dataset/ablation/masked_com_small_copaint_a6_0.json \
    --log_x0_predictions True