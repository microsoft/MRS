#!/bin/bash

# Path to data (https://github.com/zhangmozhi/mrs)
DATA_DIR=$HOME/data/mrs
# Path to mBERT (https://huggingface.co/bert-base-multilingual-cased)
MODEL_DIR=$HOME/data/bert-base-multilingual-cased  
# Path to save model
SAVED_MODEL_DIR=$HOME/mrs_model
# Path to save predictions
OUTPUT_MODEL_DIR=$HOME
# Test Language
lang=en

python retrieval_rs/models/common/sr_driver.py \
	--architecture matching_mltl_model \
	--run_mode eval \
	--train_input_dir $DATA_DIR/train \
	--valid_input_dir $DATA_DIR/valid \
	--gmr_input_dir $DATA_DIR/gmr_valid \
	--rsp_input_dir $DATA_DIR/rsp_set \
	--rsp_mapping_input_dir $DATA_DIR/rsp_map \
	--vocab_input_dir $MODEL_DIR \
	--model_input_dir $SAVED_MODEL_DIR \
	--model_output_dir $OUTPUT_DIR \
	--eval_output_dir $OUTPUT_DIR \
	--tokenizer wordpiece \
	--pretrained_model_path $MODEL_DIR \
	--load_from mbert \
	--txt_encoder_type MBert \
	--recon_loss_type SMLoss \
	--rsp_label_col -1 \
	--batch_size_infer 512 \
	--batch_size_validation 512 \
	--batch_size 512 \
	--elbo_lambda 0.9 \
	--max_msg_len 70 --max_rsp_len 30 \
	--add_matching_loss_reg 0.9 --decay_matching_loss 1.0 \
	--model_language_class multi_lingual \
	--lm_alpha_multi "0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0_0" \
	--test_langs ${lang}

# NOTE: Need to add --ja flag if test language is Japanese
python ../eval_single_ref.py $OUTPUT_DIR/${lang}_model_responses.txt
