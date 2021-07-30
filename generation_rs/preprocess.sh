# Adapted from Unicoder/generation/bash_scripts/preprocess/preprocess_{NGT,QG}.sh

########## Change these

# Path to Unicoder/generation repo (https://github.com/microsoft/Unicoder/tree/master/generation)
CODE_ROOT=$HOME/Unicoder/generation
# Path to Unicoder-xDAE model (https://onedrive.live.com/?authkey=%21AOkzIYo8pYONMb4&id=5C2C1309D09F7C6B%212278&cid=5C2C1309D09F7C6B)
MODEL_DIR=$HOME/data/Unicoder_xDAE
# Path to MRS (https://github.com/zhangmozhi/mrs)
RAW_DATA=$HOME/data/mrs
# Path to save processed dataset
DATA=$HOME/data/mrs_generation

##########

SPE_MODEL=$MODEL_DIR/sentencepiece.bpe.model
DICT=$MODEL_DIR/dict.txt

DATA_SPM=$DATA/spm
DATA_BIN=$DATA/bin
DATA_REF=$DATA/ref

# Convert MRS to Unicoder data format
python preprocess.py $RAW_DATA $DATA

mkdir -p $DATA_SPM
mkdir -p $DATA_REF

# Save references

for lg in en es de pt fr ja sv it nl ru; do
    cp ${DATA}/reddit.$lg.tgt.dev ${DATA_REF}/$lg.tgt.valid 
done


# Tokenize

for lg in en es de pt fr ja sv it nl ru; do
    for split in train dev; do
        for pair in tgt src; do
            echo $lg.$pair.$split
            python $CODE_ROOT/scripts/spm_encode.py --model $SPE_MODEL \
                --inputs ${DATA}/reddit.$lg.$pair.$split --outputs ${DATA}/spm/$lg.$split.spm.$pair
        done
    done
    for split in test; do
        for pair in src; do
            echo $lg.$pair.$split
            python $CODE_ROOT/scripts/spm_encode.py --model $SPE_MODEL \
                --inputs ${DATA}/reddit.$lg.$pair.$split --outputs ${DATA}/spm/$lg.$split.spm.$pair
        done
    done
done

# Truncate source to 512

python $CODE_ROOT/bash_scripts/preprocess/truncate_src.py --path $DATA_SPM --max_len 512

# Binarize

for lg in en es de pt fr ja sv it nl ru; do
    echo $lg
    mkdir -p $DATA_BIN/$lg
    python $CODE_ROOT/preprocess.py \
        --source-lang src \
        --target-lang tgt \
        --only-source \
        --testpref $DATA_SPM/$lg.test.spm \
        --destdir $DATA_BIN/$lg \
        --thresholdtgt 0 \
        --thresholdsrc 0 \
        --srcdict ${DICT} \
        --workers 120
done


for lg in en es de pt fr ja sv it nl ru; do
    echo $lg
    mkdir -p $DATA_BIN/$lg
    python $CODE_ROOT/preprocess.py \
        --source-lang src \
        --target-lang tgt \
        --trainpref $DATA_SPM/$lg.train.spm \
        --validpref $DATA_SPM/$lg.dev.spm \
        --destdir $DATA_BIN/$lg \
        --thresholdtgt 0 \
        --thresholdsrc 0 \
        --srcdict ${DICT} \
        --tgtdict ${DICT} \
        --workers 120
done

echo "Done!"
