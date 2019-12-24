gpu=0;
beam_size=1;
random_sampling_topk=10;
random_sampling_temp=1;

while getopts m:s:t:a:l:n:o:b:k:p:g: option
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
b) beam_size=${OPTARG};;
k) random_sampling_topk=${OPTARG};;
p) random_sampling_temp=${OPTARG};;
g) gpu=${OPTARG};;
esac
done

output_file="${output}-b${beam_size}-topk${random_sampling_topk}-topp${random_sampling_temp}-min${min_length}-max${max_length}.txt"

python translate.py \
    -beam_size $beam_size \
    -model $model \
    -src $src \
    -tgt $tgt \
    -agenda $agenda \
    -min_length $min_length \
    -max_length $max_length \
    -random_sampling_topk $random_sampling_topk \
    -random_sampling_temp $random_sampling_temp \
    -output $output_file \
    -gpu $gpu

python tools/bleu.py --gen $output_file --tgt $tgt

python gpt2/decode_text.py \
    --src $output_file \
    --dst "${output_file}.decoded"

echo "${output_file}.decoded"

# sh translate_and_view.sh -m output/now_youre_cooking_BPE/multipsa_now_youre_cooking/cooking_multipsa/checkpoints/model_step_2000.pt -s data/now_youre_cooking/test.txt.src.bpe -a data/now_youre_cooking/test.txt.agenda.bpe -t data/now_youre_cooking/test.txt.tgt.bpe -l 100 -n 150 -o generations/generations.txt -g 0 -v