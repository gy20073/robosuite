#export TF_XLA_FLAGS=--tf_xla_cpu_global_jit=/home/yeweirui/anaconda3/envs/rl/lib/python3.6/site-packages/tensorflow/compiler/xla:$TF_XLA_FLAGS=--tf_xla_cpu_global_jit

python demo_bin_packing_baselines.py \
    --alg ppo \
    --num_envs 4 \
    --control_freq 1 \
    --total_timesteps 100000 \
    --nsteps 128 \
    --save_interval 100 \
    --lr 1e-3 \
    --network mlp \
    --num_layers 2 \
    --debug 'test'
