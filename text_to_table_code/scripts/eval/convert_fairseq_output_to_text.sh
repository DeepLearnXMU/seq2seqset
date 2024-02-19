x=$1
header=$2

python $( dirname $0 )/get_hypothesis.py $x $x.hyp
python $( dirname $0 )/gpt2_decode.py $x.hyp $x.text
if [[ $header != "" ]];then
    python $( dirname $0 )/add_header.py $header $x $x.text $x.header
fi
