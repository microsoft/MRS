# Adapted from Unicoder/generation/bash_scripts/finetune/finetune_{QG,NTG}.sh

########## Change these

# Path to Unicoder/generation
CODE_ROOT=$HOME/Unicoder/generation
# Path to Unicoder-xDAE
MODEL_DIR=$HOME/data/Unicoder_xDAE
# Path to save model checkpoints
OUTPUT_DIR=$HOME/mrs_generation_model
# Path to preprocessed MRS
DATA_ROOT=$HOME/data/mrs_generation
# Training language
lg=en
# Number of GPUs
NGPU=1

##########

# Hyperparameters
lr=1e-5 # learning rate
warmup=1000 # warmup steps
max_update=5000 # max SGD steps
TBS=1024 # batch size
max_sents=4 # number of sentences to be processed at the same time
update_freq=$(($TBS/$max_sents/$NGPU))

PRETRAIN=$MODEL_DIR/checkpoint.pt
SPE=$MODEL_DIR/sentencepiece.bpe.model

DATA_BIN=$DATA_ROOT/bin
DATA_REF=$DATA_ROOT/ref

# pretraining languages; do not change
langs=af,als,am,an,ang,ar,arz,ast,az,bar,be,bg,bn,br,bs,ca,ceb,ckb,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gan,gl,gu,he,hi,hr,hu,hy,ia,id,is,it,ja,jv,ka,kk,kn,ko,ku,la,lb,lt,lv,mk,ml,mn,mr,ms,my,nds,ne,nl,nn,no,oc,pl,pt,ro,ru,scn,sco,sh,si,simple,sk,sl,sq,sr,sv,sw,ta,te,th,tl,tr,tt,uk,ur,uz,vi,war,wuu,yi,zh,zh_classical,zh_min_nan,zh_yue

task=generation_from_pretrained_bart
EXP="MRS_${lg}"
SAVE=${OUTPUT_DIR}/$EXP
mkdir -p $SAVE

SUFFIX=""
if [ ! -f $SAVE/checkpoint_last.pt ]; then
   echo "copy pretrained model to last"
   cp $PRETRAIN $SAVE/checkpoint_last.pt
   SUFFIX="$SUFFIX --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer"
fi

python $CODE_ROOT/train.py ${DATA_BIN}/${lg}  \
           --save-dir $SAVE \
           --arch mbart_base \
           --encoder-layers 12 \
           --decoder-layers 12 \
           --max-source-positions 512 \
           --max-target-positions 512 \
           --disable-validation \
           --task $task \
           --source-lang $lg \
           --target-lang $lg \
           --criterion label_smoothed_cross_entropy \
           --label-smoothing 0.2  \
           --common_eos EOS \
           --placeholder 200 \
           --dataset-impl mmap \
           --optimizer adam \
           --adam-eps 1e-06 \
           --adam-betas '(0.9, 0.98)' \
           --lr-scheduler inverse_sqrt \
           --lr $lr --min-lr -1 \
           --warmup-updates $warmup \
           --dropout 0.1 \
           --attention-dropout 0.1  \
           --weight-decay 0.01 \
           --max-sentences $max_sents \
           --update-freq $update_freq \
           --max-update $max_update \
           --save-interval-updates $max_update \
           --seed 1 \
           --log-format simple --log-interval 100 \
           --langs $langs \
           --layernorm-embedding  --ddp-backend no_c10d --fp16 \
           --freeze_layers encoder.embed_tokens \
           $SUFFIX
