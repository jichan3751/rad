export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs

# with rad crop augmentation
python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel --work_dir ./tmp \
    --action_repeat 8 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 84 \
    --agent rad_sac --frame_stack 3 --data_augs crop  \
    --save_model \
    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 200000 \
    | tee -i logs/train_rad.txt

# with no augmentation
python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel --work_dir ./tmp \
    --action_repeat 8 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 84 \
    --agent rad_sac --frame_stack 3 --data_augs no_aug  \
    --save_model \
    --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 200000\
    | tee -i logs/train_no_aug.txt
