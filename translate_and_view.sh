while getopts m:s:t:a:l:n:o:g:v option
do
case "${option}"
in
m) model=${OPTARG};;
s) src=${OPTARG};;
t) tgt=${OPTARG};;
a) agenda=${OPTARG};;
l) min_length=${OPTARG};;
n) max_length=${OPTARG};;
o) output=${OPTARG};;
v) verbose=${OPTARG};;
g) gpu=${OPTARG};;
esac
done

python translate.py \
    -beam_size 1 \
    -model $model \
    -src $src \
    -tgt $tgt \
    -agenda $agenda \
    -min_length $min_length \
    -max_length $max_length \
    -random_sampling_topk 10 \
    -output $output \
    -gpu $gpu \
    $verbose

python tools/bleu.py --gen $output --tgt $tgt

python gpt2/decode_text.py --src "${output}" --dst "${output}.decoded"

echo "${output}.decoded"
