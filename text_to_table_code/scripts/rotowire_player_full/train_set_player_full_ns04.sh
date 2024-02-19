# Run with 8 GPU
seed=${3:-"1"}
EXP=rotowire_set_player_full_ns04-$seed
DATA_PATH=data/rotowire_player_full
BART_PATH=bart.base

TOTAL_NUM_UPDATES=120000
WARMUP_UPDATES=400
LR=1e-04
MAX_TOKENS=2048
UPDATE_FREQ=1
size=base
MAX_NUM=20
INTERVAL=1
NULL_SCALE=0.4

# train
if [[ $1 == "debug" ]];then
    cmd="-m debugpy --listen 5678 --wait-for-client"
    echo "Debug mode"
fi
mkdir -p checkpoints/$EXP
python $cmd fairseq-0.12.2/fairseq_cli/train.py --num-workers 4 ${DATA_PATH}/bins \
    --user-dir set \
    --task seq2set3 \
    --arch seq2set3_bart_base \
    --max-num ${MAX_NUM} \
    --criterion seq2set3_label_smoothed_cross_entropy --assign-steps 4 --null-scale ${NULL_SCALE} \
    --save-dir checkpoints/$EXP \
    --seed $seed \
    --keep-best-checkpoints 3 --save-interval $INTERVAL --validate-interval $INTERVAL \
    --finetune-from-model ${BART_PATH}/model.pt \
    --max-tokens $MAX_TOKENS \
    --source-lang text --target-lang data \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --required-batch-size-multiple 1 \
    --label-smoothing 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --lr-scheduler inverse_sqrt --lr $LR --max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES --warmup-init-lr '1e-07' \
    --dropout 0.1 --attention-dropout 0.1 \
    --clip-norm 0.1 \
    --amp --update-freq $UPDATE_FREQ \
    --aim-repo . | tee -a logs/$EXP.log

# average checkpoints
bash scripts/eval/average_ckpt_best.sh checkpoints/$EXP
