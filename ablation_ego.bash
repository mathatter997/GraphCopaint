lr_xt_paths=(lr_x/exp_init25_decay1.0 \
            lr_x/exp_init25_decay1.002 \
            lr_x/exp_init25_decay1.004 \
            lr_x/exp_init25_decay1.006 \
            lr_x/relu_init25_start100_xT50 \
            lr_x/relu_init25_start200_xT50
            lr_x/relu_init25_start100_xT200 \
            lr_x/relu_init25_start200_xT200 \
            lr_x/relu_init25_start100_xT400 \
            lr_x/relu_init25_start200_xT400 \
            )

loss_mode="naive_inpaint"
reg_mode="naive_square"

for ((i = 0; i < ${#lr_xt_paths[@]}; i++)); do
    CUDA_VISIBLE_DEVICES="0" python inference.py --config_type ego_small  --cpu True --num_samples 32 \
        --sampler ddim --inpainter 'copaint' --num_timesteps 1000 \
        --lr_xt_path ${lr_xt_paths[$i]}.json \
        --loss_mode ${loss_mode} --reg_mode ${reg_mode} \
        --num_intervals 1 --optimization_steps 2 --tau 5 --time_travel True \
        --use_adaptive_lr_xt False \
        --checkpoint_path models/Ego_small/gnn/checkpoint_epoch_400000_t1000_psgn.pth \
        --scheduler_path models/Ego_small/scheduler_config.json \
        --output_path data/dataset/ablation/ego/output_copaint_${lr_xt_paths[$i]}.json \
        --mask_path data/dataset/ablation/ego/mask_copaint_${lr_xt_paths[$i]}.json \
        --masked_output_path data/dataset/ablation/ego/masked_output_copaint_${lr_xt_paths[$i]}.json \
        --log_x0_predictions True
done