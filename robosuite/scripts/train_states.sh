#export TF_XLA_FLAGS=--tf_xla_cpu_global_jit=/home/yeweirui/anaconda3/envs/rl/lib/python3.6/site-packages/tensorflow/compiler/xla:$TF_XLA_FLAGS=--tf_xla_cpu_global_jit

python bin_packing_baselines_states.py \
    --alg ppo \
    --num_envs 8 \
    --control_freq 1 \
    --total_timesteps 800000 \
    --nsteps 256 \
    --save_interval 100 \
    --lr 1e-4 \
    --network mlp \
    --num_layers 3 \
    --load_path '/home/yeweirui/checkpoint/openai-2020-02-09-09-47-57-250637/checkpoints/00800' \
    --debug 'more_object'
