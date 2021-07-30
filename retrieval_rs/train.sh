#!/bin/bash

########## Change these

# Path to data (https://github.com/zhangmozhi/mrs)
DATA_DIR=/mnt/Data/smartreply/reddit_15_langs
# Path to mBERT (https://huggingface.co/bert-base-multilingual-cased)
MODEL_DIR=/mnt/Data/pretrained_models/BERT/bert-base-multilingual-cased
# Path to save model
OUTPUT_DIR=/mnt/Data/smartreply/models/matching

##########

# Training languages (can be multiple languages separated by underscore: e.g., en_es)
LANG=es_pt
# Training epochs
EPOCH=20
# Batch Size
BATCH=256

python retrieval_rs/models/common/sr_driver.py \
	--architecture matching_mltl_model \
	--train_input_dir $DATA_DIR/train \
	--valid_input_dir $DATA_DIR/valid \
	--gmr_input_dir $DATA_DIR/gmr_valid \
	--rsp_input_dir $DATA_DIR/rsp_set \
	--vocab_input_dir $MODEL_DIR \
	--model_output_dir $OUTPUT_DIR \
	--tokenizer wordpiece \
	--pretrained_model_path $MODEL_DIR \
	--load_from mbert \
	--txt_encoder_type MBert \
	--recon_loss_type SMLoss \
	--rsp_label_col -1 \
	--batch_size_infer $BATCH \
	--batch_size_validation $BATCH \
	--batch_size $BATCH \
	--validation_batches 50 \
	--infer_batches 50 \
	--save_freq 5000 \
	--validation_freq 5 \
	--steps_per_print 200 \
	--train_msg_encoder --train_rsp_encoder \
	--elbo_lambda 0.9 \
	--sweep_lm_window 0.05 --sweep_lm --sweep_lm_span 1 \
	--lm_alpha 0.5 \
	--optimizer adam --fp16 --learning_rate 0.000001 --decay_step 5000 --decay_rate 0.999 --warmup_proportion 0.001 --loss_scale 0 \
	--max_msg_len 70 --max_rsp_len 30 \
	--add_matching_loss_reg 0.9 --decay_matching_loss 1.0 \
	--freeze_layers embedding#transformer_0_1 \
	--model_language_class multi_lingual \
	--train_langs $LANG \
	--test_langs $LANG \
	--sample_languages_uniformly \
	--lm_alpha_multi "0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5_0.5" \
	--max_epochs $EPOCH \
	--truncate
	--fp16

# NOTE: Distributed training is also supported. Use the following command for multiple GPUs.
# NGPU=8 
# python -m torch.distributed.launch --nproc_per_node=$NGPU retrieval_rs/models/common/sr_driver.py \
	# --distributed_data_parallel \
	# ...
