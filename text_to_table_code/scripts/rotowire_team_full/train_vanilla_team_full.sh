# Run with 8 GPU
seed=${3:-"1"}
EXP=rotowire_vanilla_team_full-$seed
DATA_PATH=data/rotowire_team_full
BART_PATH=bart.base

TOTAL_NUM_UPDATES="8000"
WARMUP_UPDATES="400"
LR="3e-05"
MAX_TOKENS="4096"
UPDATE_FREQ="1" # simulate 8 GPUs
INTERVAL="10"
size="base"

# train
mkdir -p checkpoints/$EXP
python fairseq-0.12.2/fairseq_cli/train.py --num-workers 4 ${DATA_PATH}/bins \
    --save-dir checkpoints/$EXP \
    --seed $seed \
    --keep-best-checkpoints 3 --no-epoch-checkpoints --save-interval $INTERVAL --validate-interval $INTERVAL \
    --finetune-from-model ${BART_PATH}/model.pt \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang text --target-lang data \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --required-batch-size-multiple 1 \
    --arch bart_${size} \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler inverse_sqrt --lr $LR --max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES --warmup-init-lr '1e-07' \
    --amp --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --aim-repo . | tee -a logs/$EXP.log

# average checkpoints
bash scripts/eval/average_ckpt_best.sh checkpoints/$EXP
