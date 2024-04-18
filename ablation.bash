# # ablation 1

# loss_mode="naive_inpaint"
# reg_mode="naive_square"
# lr_xt_list=(0.0025 0.004 0.01 0.1 1)
# for ((i = 0; i < ${#lr_xt_list[@]}; i++)); do
#     CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu False --num_samples 32 \
#     --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
#     --lr_xt ${lr_xt_list[$i]} \
#     --loss_mode $loss_mode --reg_mode $reg_mode \
#     --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
#     --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
#     --scheduler_path models/Community_small/scheduler_config.json \
#     --output_path data/dataset/ablation/output_com_small_copaint_a1_${i}.json \
#     --mask_path data/dataset/ablation/mask_com_small_copaint_a1_${i}.json \
#     --masked_output_path data/dataset/ablation/masked_com_small_copaint_a1_${i}.json \
#     --log_x0_predictions True
# done

# # # ablation 2

# loss_mode="naive_inpaint"
# reg_mode="naive_square"
# tau_list=(1 2 5 10)
# for ((i = 0; i < ${#lr_xt_list[@]}; i++)); do
#     CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu False --num_samples 32 \
#     --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
#     --loss_mode $loss_mode --reg_mode $reg_mode \
#     --num_intervals 1 --optimization_steps 2 --tau ${tau_list[$i]} --time_travel True \
#     --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
#     --scheduler_path models/Community_small/scheduler_config.json \
#     --output_path data/dataset/ablation/output_com_small_copaint_a2_${i}.json \
#     --mask_path data/dataset/ablation/mask_com_small_copaint_a2_${i}.json \
#     --masked_output_path data/dataset/ablation/masked_com_small_copaint_a2_${i}.json \
#     --log_x0_predictions True
# done

# ablation 3

# loss_mode="naive_inpaint"
# reg_mode="naive_square"
# lr_xt_decay_list=(1.0 1.01 1.02 1.03)
# for ((i = 0; i < ${#lr_xt_decay_list[@]}; i++)); do
#     CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu False --num_samples 32 \
#     --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
#     --loss_mode $loss_mode --reg_mode $reg_mode \
#     --lr_xt_decay ${lr_xt_decay_list[$i]} \
#     --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
#     --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
#     --scheduler_path models/Community_small/scheduler_config.json \
#     --output_path data/dataset/ablation/output_com_small_copaint_a3_${i}.json \
#     --mask_path data/dataset/ablation/mask_com_small_copaint_a3_${i}.json \
#     --masked_output_path data/dataset/ablation/masked_com_small_copaint_a3_${i}.json \
#     --log_x0_predictions True
# done

# # # ablation 4

# loss_mode="inpaint"
# reg_mode="square"
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu False --num_samples 32 \
#     --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
#     --loss_mode $loss_mode --reg_mode $reg_mode \
#     --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
#     --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
#     --scheduler_path models/Community_small/scheduler_config.json \
#     --output_path data/dataset/ablation/output_com_small_copaint_a4_0.json \
#     --mask_path data/dataset/ablation/mask_com_small_copaint_a4_0.json \
#     --masked_output_path data/dataset/ablation/masked_com_small_copaint_a4_0.json \
#     --log_x0_predictions True

# # ablation 5

# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu False --num_samples 32 \
#     --sampler ddim --inpainter 'repaint' --num_timesteps 1000 \
#     --loss_mode $loss_mode --reg_mode $reg_mode \
#     --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
#     --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
#     --scheduler_path models/Community_small/scheduler_config.json \
#     --output_path data/dataset/ablation/output_com_small_copaint_a5_0.json \
#     --mask_path data/dataset/ablation/mask_com_small_copaint_a5_0.json \
#     --masked_output_path data/dataset/ablation/masked_com_small_copaint_a5_0.json \
#     --log_x0_predictions True


loss_mode="naive_inpaint"
reg_mode="naive_square"
lr_xt_decay_list=(1.0 1.01 1.02 1.03 1.04 1.05)
for ((i = 0; i < ${#lr_xt_decay_list[@]}; i++)); do
    CUDA_VISIBLE_DEVICES="2" python inference.py --config_type community_small_smooth  --cpu False --num_samples 32 \
        --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
        --loss_mode $loss_mode --reg_mode $reg_mode \
        --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
        --checkpoint_path models/Community_small_smooth/gnn/checkpoint_epoch_20000_t1000_psgn.pth \
        --scheduler_path models/Community_small_smooth/scheduler_config.json \
        --output_path data/dataset/ablation/output_com_small_smooth_copaint_a1_${i}.json \
        --mask_path data/dataset/ablation/mask_com_small_smooth_copaint_a1_${i}.json \
        --masked_output_path data/dataset/ablation/masked_com_small_smooth_copaint_a1_${i}.json \
        --log_x0_predictions True
done

loss_mode="naive_inpaint"
reg_mode="naive_square"
lr_xt_decay_list=(1.0 1.01 1.02 1.03 1.04 1.05)
for ((i = 0; i < ${#lr_xt_decay_list[@]}; i++)); do
    CUDA_VISIBLE_DEVICES="2" python inference.py --config_type community_small_smooth  --cpu False --num_samples 32 \
        --sampler ddpm --inpainter 'copaint' --num_timesteps 1000 \
        --loss_mode $loss_mode --reg_mode $reg_mode \
        --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
        --checkpoint_path models/Community_small_smooth/gnn/checkpoint_epoch_20000_t1000_psgn.pth \
        --scheduler_path models/Community_small_smooth/scheduler_config.json \
        --output_path data/dataset/ablation/output_com_small_smooth_copaint_a2_${i}.json \
        --mask_path data/dataset/ablation/mask_com_small_smooth_copaint_a2_${i}.json \
        --masked_output_path data/dataset/ablation/masked_com_small_smooth_copaint_a2_${i}.json \
        --log_x0_predictions True
done

loss_mode="naive_inpaint"
reg_mode="naive_square"
CUDA_VISIBLE_DEVICES="2" python inference.py --config_type community_small_smooth  --cpu False --num_samples 32 \
    --sampler ddpm --inpainter 'repaint' --num_timesteps 1000 \
    --loss_mode $loss_mode --reg_mode $reg_mode \
    --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
    --checkpoint_path models/Community_small_smooth/gnn/checkpoint_epoch_20000_t1000_psgn.pth \
    --scheduler_path models/Community_small_smooth/scheduler_config.json \
    --output_path data/dataset/ablation/output_com_small_smooth_copaint_a3.json \
    --mask_path data/dataset/ablation/mask_com_small_smooth_copaint_a3.json \
    --masked_output_path data/dataset/ablation/masked_com_small_smooth_copaint_a3.json \
    --log_x0_predictions True
done

loss_mode="naive_inpaint"
reg_mode="naive_square"
CUDA_VISIBLE_DEVICES="2" python inference.py --config_type community_small_smooth  --cpu False --num_samples 32 \
    --sampler ddim --inpainter 'repaint' --num_timesteps 1000 \
    --loss_mode $loss_mode --reg_mode $reg_mode \
    --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
    --checkpoint_path models/Community_small_smooth/gnn/checkpoint_epoch_20000_t1000_psgn.pth \
    --scheduler_path models/Community_small_smooth/scheduler_config.json \
    --output_path data/dataset/ablation/output_com_small_smooth_copaint_a4.json \
    --mask_path data/dataset/ablation/mask_com_small_smooth_copaint_a4.json \
    --masked_output_path data/dataset/ablation/masked_com_small_smooth_copaint_a4.json \
    --log_x0_predictions True
done




