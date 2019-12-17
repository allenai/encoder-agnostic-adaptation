# Encoder-Agnostic Adaptation for Conditional Language Generation

This repo is a fork of [encoder-agnostic](https://github.com/harvardnlp/encoder-agnostic-adaptation) implementing [Encoder-Agnostic Adaptation for Conditional Language Generation](https://arxiv.org/abs/1908.06938), Zachary M. Ziegler, Luke Melas-Kyriazi, Sebastian Gehrmann and Alexander M. Rush. It extends [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

A more elaborated readme about the encoder-agnostic model, checkout the original github repo.

This code was tested with `pytorch 1.0.1`. See requirements.txt for a complete list of dependencies.

## Download GPT2 weights

`cd gpt2 && python download_model.py 124M`

## Data

You can find the BPEized data the original paper used in the experiments [here](https://drive.google.com/file/d/1Z6AdOr2MtWlN7sYRTMibzAcghBjSBzZK/view?usp=sharing). 

### Your Data
I expect using agenda items. I you do not, Just ignore all agenda related notes.

You should create 9 files. train.txt.src, train.txt.tgt, train.txt.agenda, valid.txt.src, valid.txt.tgt, valid.txt.agenda, test.txt.src, test.txt.tgt, test.txt.agenda.
If your testbed is the tacred dataset, you can create corresponding datafiles using `python scripts/create_datafiles.py`. Notice the possible arguments the script takes.
For the "now you're cooking" dataset, download it from [here](http://homes.cs.washington.edu/~antoineb/datasets/now_youre_cooking.tar.gz) and use:
```
python create_recipes_datafiles.py --dataset ...now_youre_cooking/train --output data/now_youre_cooking/train
python create_recipes_datafiles.py --dataset ...now_youre_cooking/dev --output data/now_youre_cooking/dev
python create_recipes_datafiles.py --dataset ...now_youre_cooking/test --output data/now_youre_cooking/test
```

#### BPE Your Data

To run any of these models with your own data you should first BPEize it with `python gpt2/encode_text.py <filename>`. Before training the raw data is preprocessed into binary data shards with the commands below.

Set up a configuration file similiarly to the ones you can find in `config` folder.

Then you need to preprocess you data. example below.

## Tacred

### Preprocess

`python preprocess.py -train_src .../train.txt.src.bpe -train_tgt .../train.txt.tgt.bpe -train_agenda .../train.txt.agenda.bpe -valid_src .../valid.txt.src.bpe -valid_tgt .../valid.txt.tgt.bpe -valid_agenda .../valid.txt.agenda.bpe -save_data .../BPE -tgt_seq_length_trunc 100 -src_vocab gpt2/vocab.txt -tgt_vocab gpt2/vocab.txt -agenda_vocab gpt2/vocab.txt -fixed_vocab`

### Train
Currently I'm experimenting with Pseudo self attention.

**Pseudo self attention**: `python train.py -config config/config_file.yml -run_name run_name -gpt2_params_path gpt2/models/124M/ -gpt2_init_embanddec -include_agenda > log.txt 2>&1 &`

### Generation

Generation is performed via random sampling.

`python translate.py -model .../checkpoints/model_step_1000.pt -src .../src.bpe -min_length 1 -max_length 40 -beam_size 1 -random_sampling_topk 10 -random_sampling_temp 1 -output generations/generations.txt -v`

Then, you would probably want to decode the outputs now:

`python gpt2/decode_text.py --src generations/generations.txt --dst generations/generations.txt.bpe.decoded`