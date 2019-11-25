while getopts m:s:b:l:n:k:p:o:v option
do
case "${option}"
in
m) model=${OPTARG};;
s) src=${OPTARG};;
b) beam_size=${OPTARG};;
l) min_length=${OPTARG};;
n) max_length=${OPTARG};;
k) random_sampling_topk=${OPTARG};;
p) random_sampling_temp=${OPTARG};;
o) output=${OPTARG};;
v) verbose=${OPTARG};;
esac
done

python translate.py \
    -beam_size $beam_size \
    -model $model \
    -src $src \
    -min_length $min_length \
    -max_length $max_length \
    -random_sampling_topk $random_sampling_topk \
    -random_sampling_temp $random_sampling_temp \
    -output "${output}-b${beam_size}-topk${random_sampling_topk}-topp${random_sampling_temp}" \
    $verbose

python gpt2/decode_text.py \
    --src "${output}-b${beam_size}-topk${random_sampling_topk}-topp${random_sampling_temp}" \
    --dst "${output}-b${beam_size}-topk${random_sampling_topk}-topp${random_sampling_temp}.decoded"


head "${output}-b${beam_size}-topk${random_sampling_topk}-topp${random_sampling_temp}.decoded"
echo ""
echo "${output}-b${beam_size}-topk${random_sampling_topk}-topp${random_sampling_temp}.decoded"
