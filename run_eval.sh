# run evaluation loop
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2

mkdir -p logs
declare -a augs=("no_aug" "crop")
declare -a cors=(
    "no_cor"
    "gaussian_noise"
    "shot_noise"
    "impulse_noise"
    "speckle_noise"
    "gaussian_blur"
    "glass_blur"
    "defocus_blur"
    "motion_blur"
    "zoom_blur"
    "fog"
    "frost"
    "snow"
    "spatter"
    "contrast"
    "brightness"
    "saturate"
    "jpeg_compression"
    "pixelate"
)
declare -a sevs=(1 2 3 4 5)

# test
# declare -a cors=( "no_cor" "gaussian_noise")
# declare -a sevs=(1 2)


declare -a augs=("no_aug")
declare -a cors=("gaussian_noise")
declare -a sevs=(1 2 3 4)

trap 'killall' INT

killall() {
    trap '' INT TERM     # ignore INT and TERM while shutting down
    echo "**** Shutting down... ****"     # added double quotes
    kill -TERM 0         # fixed order, send TERM not INT
    wait
    echo DONE
}

for aug in "${augs[@]}"
do
    for cor in "${cors[@]}"
    do
        for sev in "${sevs[@]}"
        do
            exp_log="logs/eval_${aug}_${cor}${sev}.txt"
            echo "log: $exp_log"

            python eval_corruptions.py \
                --cor_func $cor --cor_sev $sev \
                --domain_name cartpole \
                --task_name swingup \
                --encoder_type pixel --work_dir ./tmp \
                --action_repeat 8 --num_eval_episodes 10 \
                --pre_transform_image_size 100 --image_size 84 \
                --agent rad_sac --frame_stack 3 --data_augs no_aug  \
                --save_model \
                --seed 23 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 200000 \
                > $exp_log & 
        done
        wait
    done
done

