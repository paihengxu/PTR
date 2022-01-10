export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 run_prompt.py \
--data_dir ../datasets/semeval \
--output_dir ../results/semeval \
--model_type bert \
--model_name_or_path bert-base-cased \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 1 \
--max_seq_length 512 \
--warmup_steps 500 \
--learning_rate 3e-5 \
--learning_rate_for_new_token 1e-5 \
--num_train_epochs 5 \
--weight_decay 1e-2 \
--adam_epsilon 1e-6 \
--temps temp.txt
