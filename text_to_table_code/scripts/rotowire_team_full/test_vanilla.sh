DATA_PATH=data/rotowire_team_full
ckpt=${2:-"checkpoints/rotowire_vanilla_team_full-1/checkpoint_best.pt"}

export PYTHONPATH=.

if [[ $1 == "debug" ]];then
  cmd="-m debugpy --listen 5678 --wait-for-client"
fi
# python $cmd fairseq-0.12.2/fairseq_cli/generate.py ${DATA_PATH}/bins --path $ckpt --beam 5 --remove-bpe --max-len-b 1024 > $ckpt.test_vanilla.out 
# bash scripts/eval/convert_fairseq_output_to_text.sh $ckpt.test_vanilla.out

for table in Team; do
  printf "$table table wrong format:\n"
  python scripts/eval/calc_data_wrong_format_ratio.py $ckpt.test_vanilla.out.text ${DATA_PATH}/test.data --row-header --col-header 
  for metric in E c BS-scaled; do
    printf "$table table $metric metric:\n"
    python scripts/eval/calc_data_f_score.py $ckpt.test_vanilla.out.text ${DATA_PATH}/test.data --row-header --col-header --metric $metric
  done
done