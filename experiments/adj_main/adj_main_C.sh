#!/usr/bin/env bash
# from adj_new_B.1.sh

ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

MYDIR=logs/${ID}
mkdir -p ${MYDIR}
cp ${ABS_PATH} ${MYDIR}

CUDA_VISIBLE_DEVICES=5 \
python -u main.py \
--experiment_id ${ID} \
--data_name scan \
--train_file data_adj/train_hard.txt \
--test_file data_adj/test_hard.txt \
--model_name rand_reg \
--random_seed 10 \
--batch_size 64 \
--switch_temperature 0.1 \
--attention_temperature 1 \
--num_units 16 \
--epochs 5000 \
--learning_rate 0.01 \
--max_gradient_norm 1.0 \
--use_input_length \
--use_embedding \
--embedding_size 64 \
--bidirectional_encoder \
--random_batch \
--decay_steps 100 \
--remove_switch \
--content_noise \
--function_noise \
--content_noise_coe 0.1 \
--noise_weight 0.3 \
--sample_wise_content_noise \
--masked_attention \
--random_random \
| tee ${MYDIR}/log.txt

python attention_visualization.py \
--hide_switch \
--experiment_id ${ID}