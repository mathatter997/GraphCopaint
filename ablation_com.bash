# # ablation 1

# loss_mode="naive_inpaint"
# reg_mode="naive_square"
# lr_xt_list=(0.0025 0.004 0.01 0.1 1)
# for ((i = 0; i < ${#lr_xt_list[@]}; i++)); do
#     CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 32 \
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


 # --lr_xt 0.0025 --lr_xt_decay 1.01 \

# lr_xt_paths=(lr_x/exp_init25_decay1.0.json \
#             lr_x/exp_init25_decay1.002.json \
#             lr_x/exp_init25_decay1.004.json \
#             lr_x/exp_init25_decay1.006.json \
#             lr_x/relu_init25_start100_xT50.json \
#             lr_x/relu_init25_start200_xT50.json
#             )
# lr_xt_paths=(lr_x/relu_init25_start100_xT200.json \
#             lr_x/relu_init25_start200_xT200.json \
#             lr_x/relu_init25_start100_xT400.json \
#             lr_x/relu_init25_start200_xT400.json \
#             )
# lr_xt_paths=(lr_x/const_init10.json \
#             lr_x/const_init25.json \
#             lr_x/const_init50.json \
#             lr_x/const_init100.json \
#             )

# loss_mode="naive_inpaint"
# reg_mode="naive_square"
# lr_xt_path=lr_x/exp_init25_decay1.006.json
# opt_num_paths=(opt_num/const_init2.json \
#             opt_num/const_init5.json \
#             opt_num/const_init10.json \
#             opt_num/relu_init2_start100_xT10.json \
#             opt_num/relu_init2_start100_xT10.json \
#             )
# for ((i = 0; i < ${#opt_num_paths[@]}; i++)); do
#     CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 1 \
#         --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
#         --lr_xt_path ${lr_xt_path} --opt_num_path ${opt_num_paths[$i]} \
#         --loss_mode ${loss_mode} --reg_mode ${reg_mode} \
#         --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
#         --use_adaptive_lr_xt False \
#         --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
#         --scheduler_path models/Community_small/scheduler_config.json \
#         --output_path data/dataset/ablation/output_com_small_copaint_a11_${i}.json \
#         --mask_path data/dataset/ablation/mask_com_small_copaint_a11_${i}.json \
#         --masked_output_path data/dataset/ablation/masked_com_small_copaint_a11_${i}.json \
#         --log_x0_predictions False
# done

loss_mode="naive_inpaint"
reg_mode="naive_square"
i=0
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 32 \
#         --sampler ddim --inpainter 'repaint' --num_timesteps 1000 \
#         --loss_mode ${loss_mode} --reg_mode ${reg_mode} \
#         --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
#         --use_adaptive_lr_xt False \
#         --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
#         --scheduler_path models/Community_small/scheduler_config.json \
#         --output_path data/dataset/ablation/output_com_small_copaint_a13_${i}.json \
#         --mask_path data/dataset/ablation/mask_com_small_copaint_a13_${i}.json \
#         --masked_output_path data/dataset/ablation/masked_com_small_copaint_a13_${i}.json \
#         --log_x0_predictions True

# i=1
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 32 \
#         --sampler ddpm --inpainter 'repaint' --num_timesteps 1000 \
#         --loss_mode ${loss_mode} --reg_mode ${reg_mode} \
#         --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
#         --use_adaptive_lr_xt False \
#         --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
#         --scheduler_path models/Community_small/scheduler_config.json \
#         --output_path data/dataset/ablation/output_com_small_copaint_a13_${i}.json \
#         --mask_path data/dataset/ablation/mask_com_small_copaint_a13_${i}.json \
#         --masked_output_path data/dataset/ablation/masked_com_small_copaint_a13_${i}.json \
#         --log_x0_predictions True

# i=2
# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 32 \
#         --sampler vpsde --inpainter 'repaint' --num_timesteps 1000 \
#         --loss_mode ${loss_mode} --reg_mode ${reg_mode} \
#         --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
#         --use_adaptive_lr_xt False \
#         --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
#         --scheduler_path models/Community_small/scheduler_config.json \
#         --output_path data/dataset/ablation/output_com_small_copaint_a13_${i}.json \
#         --mask_path data/dataset/ablation/mask_com_small_copaint_a13_${i}.json \
#         --masked_output_path data/dataset/ablation/masked_com_small_copaint_a13_${i}.json \
#         --log_x0_predictions True



# CUDA_VISIBLE_DEVICES="0" python inference.py --config_type community_small  --cpu True --num_samples 1 \
#     --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
#     --loss_mode $loss_mode --reg_mode $reg_mode --lr_xt_decay 1.006 \
#     --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
#     --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
#     --scheduler_path models/Community_small/scheduler_config.json \
#     --output_path data/dataset/ablation/output_com_small_copaint_b5_0.json \
#     --mask_path data/dataset/ablation/mask_com_small_copaint_b5_0.json \
#     --masked_output_path data/dataset/ablation/masked_com_small_copaint_b5_0.json \
#     --log_x0_predictions False




 # --lr_xt 0.0025 --lr_xt_decay 1.01 \

# lr_xt_path=lr_x/exp_init25_decay1.006.json
# opt_num_paths=(opt_num/const_init2.json \
#             opt_num/const_init5.json \
#             opt_num/const_init10.json \
#             opt_num/relu_init2_start100_xT10.json \
#             opt_num/relu_init2_start100_xT10.json \
#             )

lr_xt_paths=(lr_x/exp_init25_decay1.0.json \
            lr_x/exp_init25_decay1.002.json \
            lr_x/exp_init25_decay1.004.json \
            lr_x/exp_init25_decay1.006.json \
            lr_x/relu_init25_start100_xT50.json \
            lr_x/relu_init25_start200_xT50.json
            lr_x/relu_init25_start100_xT200.json \
            lr_x/relu_init25_start200_xT200.json \
            lr_x/relu_init25_start100_xT400.json \
            lr_x/relu_init25_start200_xT400.json \
            )

loss_mode="naive_inpaint"
reg_mode="naive_square"

for ((i = 0; i < ${#lr_xt_paths[@]}; i++)); do
    CUDA_VISIBLE_DEVICES="1" python inference.py --config_type community_small  --cpu False --num_samples 32 \
        --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
        --lr_xt_path ${lr_xt_paths[$i]} \
        --loss_mode ${loss_mode} --reg_mode ${reg_mode} \
        --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
        --use_adaptive_lr_xt False \
        --checkpoint_path models/Community_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
        --scheduler_path models/Community_small/scheduler_config.json \
        --output_path data/dataset/ablation/com/output_copaint_${lr_xt_paths[$i]}.json \
        --mask_path data/dataset/ablation/com/mask_copaint_${lr_xt_paths[$i]}.json \
        --masked_output_path data/dataset/ablation/com/masked_output_copaint_${lr_xt_paths[$i]}.json \
        --log_x0_predictions True
done




