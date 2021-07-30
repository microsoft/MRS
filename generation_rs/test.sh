########## Change these

# Test language
lg=en
# Path to Unicoder/generation
CODE_ROOT=$HOME/Unicoder/generation
# Path to sentencepiece file
SPE=$HOME/data/Unicoder_xDAE/sentencepiece.bpe.model
# Path to save predictions
OUTPUT_DIR=$HOME/mrs_generation_output
# Path to preprocessed MRS
DATA_ROOT=$HOME/data/mrs_generation
# Path to model checkpoint
model=$HOME/mrs_generation_model/MRS_en/checkpoint_last.pt
# Evaluation split (train/valid/test)
SPLIT=valid

##########

task=generation_from_pretrained_bart
gid=0
BEAM=5  # beam size
DATA_BIN=$DATA_ROOT/bin

bash ${CODE_ROOT}/evaluation/generate_single.sh $task $gid $lg $model $SPE $OUTPUT_DIR $SPLIT $DATA_BIN $BEAM $CODE_ROOT
echo "decoding done!"

python process_output.py $DATA_ROOT/reddit.${lg}.src.dev $DATA_ROOT/reddit.${lg}.tgt.dev $OUTPUT_DIR/${lg}_src-tgt > $OUTPUT_DIR/${lg}_preds.${SPLIT}
python ../eval.py $OUTPUT_DIR/${lg}_preds.${SPLIT}
