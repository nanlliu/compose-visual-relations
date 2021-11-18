# exp is the name of the folder in which checkpoints will be saved
# can be easily trained on iGibson by using the --dataset igibson
python train.py --cond --dataset=clevr --exp=clevr_ebm --batch_size=10 --step_lr=300 \
--num_steps=60 --kl --gpus=1 --nodes=1 --filter_dim=128 --im_size=128 --self_attn \
--multiscale --norm --spec_norm --slurm --lr=1e-4 --cuda --replay_batch \
--numpy_data_path ./data/clevr_training_data.npz