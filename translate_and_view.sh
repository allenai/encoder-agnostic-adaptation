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

# sh translate_and_view.sh -m output/now_youre_cooking_BPE/multipsa_now_youre_cooking/cooking_multipsa/checkpoints/model_step_2000.pt -s data/now_youre_cooking/test.txt.src.bpe -a data/now_youre_cooking/test.txt.agenda.bpe -t data/now_youre_cooking/test.txt.tgt.bpe -l 100 -n 150 -o generations/generations.txt -g 0 -v