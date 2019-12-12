#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import sys
import gc
import torch
from functools import partial

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid', 'vocab']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)


def build_save_dataset(corpus_type, fields, readers_list, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src = opt.train_src
        agenda = opt.train_agenda
        tgt = opt.train_tgt
        pointers_file = opt.pointers_file
    else:
        src = opt.valid_src
        agenda = opt.valid_agenda
        tgt = opt.valid_tgt
        pointers_file = None

    logger.info("Reading source and target files: %s %s." % (src, tgt))
    
    tgt_shards = split_corpus(tgt, opt.shard_size)
    def create_src_shards(src, opt):
        if opt.data_type == 'imgvec':
            assert opt.shard_size <= 0
            return [src]
        elif opt.data_type == 'none':
            return [None]*99999
        else:
            return split_corpus(src, opt.shard_size)

    src_shards = create_src_shards(src, opt)
    agenda_shards = create_src_shards(agenda, opt)

    if not agenda:
        shards = zip(src_shards, tgt_shards)
    else:
        shards = zip(src_shards, agenda_shards, tgt_shards)

    dataset_paths = []
    if (corpus_type == "train" or opt.filter_valid) and tgt is not None:
        filter_pred = partial(
            inputters.filter_example, use_src_len=opt.data_type == "text",
            max_src_len=opt.src_seq_length, max_tgt_len=opt.tgt_seq_length)
    else:
        filter_pred = None

    for i, flat_shard in enumerate(shards):
        if not agenda:
            src_shard, tgt_shard = flat_shard
        else:
            src_shard, agenda_shard, tgt_shard = flat_shard
        assert opt.data_type in ['imgvec', 'none'] or len(src_shard) == len(tgt_shard)
        logger.info("Building shard %d." % i)
        src_reader, tgt_reader = readers_list[0], readers_list[1]
        if agenda:
            readers = readers_list
            data = ([("src", src_shard), ("tgt", tgt_shard), ("agenda", agenda_shard)])
            dirs = [opt.src_dir, None, None]
        elif src_reader and tgt_reader:
            readers = [src_reader, tgt_reader]
            data = ([("src", src_shard), ("tgt", tgt_shard)])
            dirs = [opt.src_dir, None]
        elif src_reader and not tgt_reader:
            readers = [src_reader]
            data = ([("src", src_shard)])
            dirs = [opt.src_dir]
        elif not src_reader and tgt_reader:
            readers = [tgt_reader]
            data = ([("tgt", tgt_shard)])
            dirs = [None] 

        dataset = inputters.Dataset(
            fields,
            readers=readers,
            data=data,
            dirs=dirs,
            sort_key=inputters.str2sortkey[opt.data_type],
            filter_pred=filter_pred,
            pointers_file=pointers_file
        )

        data_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, corpus_type, i)
        dataset_paths.append(data_path)

        logger.info(" * saving %sth %s data shard to %s."
                    % (i, corpus_type, data_path))

        dataset.save(data_path)

        del dataset.examples
        gc.collect()
        del dataset
        gc.collect()

    return dataset_paths


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency,
        opt.agenda_vocab, opt.agenda_vocab_size, opt.agenda_words_min_frequency,
        fixed_vocab=opt.fixed_vocab,
        free_src=opt.free_src, free_tgt=opt.free_tgt,
        vocab_size_multiple=opt.vocab_size_multiple
    )

    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def main(opt):
    # import random
    # if opt.shuffle == 1:
    #     for src_path, tgt_path in [(opt.train_src, opt.train_tgt), (opt.valid_src, opt.valid_tgt)]:
    #         with open(src_path, 'r') as f: src_lines = f.readlines()
    #         with open(tgt_path, 'r') as f: tgt_lines = f.readlines()
    #         with open(src_path+".unshuffled", 'w') as f: f.write(''.join(src_lines))
    #         with open(tgt_path+".unshuffled", 'w') as f: f.write(''.join(tgt_lines))
    #         combined = list(zip(src_lines, tgt_lines))
    #         random.shuffle(combined)
    #         src_lines[:], tgt_lines[:] = zip(*combined)
    #         with open(src_path, 'w') as f: f.write(''.join(src_lines))
    #         with open(tgt_path, 'w') as f: f.write(''.join(tgt_lines))

    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)
    check_existing_pt_files(opt)

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = count_features(opt.train_src) if opt.data_type == 'text' \
        else 0
    tgt_nfeats = count_features(opt.train_tgt)  # tgt always text so far
    agenda_nfeats = count_features(opt.train_agenda)  # agenda always text so far
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    
    if opt.fixed_vocab:
        tgt_bos = '<|endoftext|>'
        tgt_eos = '\u0120GDDR'
        tgt_pad = '\u0120SHALL'
        tgt_unk = '\u0120RELE'

        if opt.no_spec_src:
            src_pad = None
            src_unk = None
        elif opt.free_src:
            src_pad = '<blank>'
            src_unk='<unk>'
        else:
            src_pad = '\u0120SHALL'
            src_unk = '\u0120RELE'

    else:
        tgt_bos='<s>'
        tgt_eos='</s>'
        tgt_pad = '<blank>'
        tgt_unk='<unk>'
        src_pad = '<blank>'
        src_unk='<unk>'

    fields = inputters.get_fields(
        opt.data_type,
        src_nfeats,
        tgt_nfeats,
        agenda_nfeats,
        dynamic_dict=opt.dynamic_dict,
        src_truncate=opt.src_seq_length_trunc,
        tgt_truncate=opt.tgt_seq_length_trunc,
        src_pad=src_pad,
        src_unk=src_unk,
        tgt_pad=tgt_pad,
        tgt_unk=tgt_unk,
        tgt_bos=tgt_bos,
        tgt_eos=tgt_eos,
        include_ptrs=opt.pointers_file is not None,
        include_agenda=opt.train_agenda or opt.valid_agenda)
    
    if opt.data_type == 'none':
        readers = [None]
    else:
        readers = [inputters.str2reader[opt.data_type].from_opt(opt)]
    readers.append(inputters.str2reader["text"].from_opt(opt))
    if opt.train_agenda or opt.valid_agenda:
        readers.append(inputters.str2reader["text"].from_opt(opt))

    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset(
        'train', fields, readers, opt)

    if (opt.valid_src or opt.data_type == 'none') and opt.valid_tgt:
        logger.info("Building & saving validation data...")
        build_save_dataset('valid', fields, readers, opt)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)


def _get_parser():
    parser = ArgumentParser(description='preprocess.py')

    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
