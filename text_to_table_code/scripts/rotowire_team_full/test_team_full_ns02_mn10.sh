export PYTHONPATH=.
DATA_PATH=data/rotowire_team_full
EXP=rotowire_set_team_full_ns02_mn10-1
LOG=${EXP}.log
idx=7
ckpt="checkpoints/${EXP}/checkpoint${idx}.pt"

# python $cmd fairseq-0.12.2/fairseq_cli/generate.py ${DATA_PATH}/bins \
#     --path $ckpt \
#     --beam 1 --header-beam 5 \
#     --max-tokens 4096 \
#     --max-len-a 0 --max-len-b 100 \
#     --user-dir set \
#     --max-num 10 --nullpen 0 --column-num 0 \
#     --skip-invalid-size-inputs-valid-test \
#     --task seq2set3 > $ckpt.test_set.out
# bash scripts/eval/convert_fairseq_output_to_text.sh $ckpt.test_set.out

printf "Wrong format:\n" | tee -a $LOG
python scripts/eval/calc_data_wrong_format_ratio.py $ckpt.test_set.out.text ${DATA_PATH}/test.data --row-header --col-header | tee -a $LOG
for metric in E c BS-scaled; do
    printf "$metric metric:\n"
    python scripts/eval/calc_data_f_score.py $ckpt.test_set.out.text ${DATA_PATH}/test.data --row-header --col-header --metric $metric | tee -a $LOG
done