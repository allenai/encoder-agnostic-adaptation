#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat
import os
import numpy as np
import json

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

def constraint_iter_func(f_iter):
    all_tags = []
    for json_line in f_iter:
        data = json.loads(json_line)
        words = data['words']
        probs = [p[1] for p in data['class_probabilities'][:len(words)]]
        tags = [1 if p > opt.bu_threshold else 0 for p in probs]
        all_tags.append(tags)
        #print(len(words), len(data['class_probabilities']))
        #all_tags.append(words)
    return all_tags


def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    if opt.constraint_file:
        tag_shards = split_corpus(opt.constraint_file, opt.shard_size, iter_func=constraint_iter_func, binary=False)

    translator = build_translator(opt, report_score=True)

    def create_src_shards(path, opt, binary=True):
        if opt.data_type == 'imgvec':
            assert opt.shard_size <= 0
            return [path]
        else:
            if opt.data_type == 'none':
                return [None]*99999
            else:
                return split_corpus(path, opt.shard_size, binary=binary)

    src_shards = create_src_shards(opt.src, opt)
    if opt.agenda:
        agenda_shards = create_src_shards(opt.agenda, opt, False)

    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
 
    if not opt.agenda:
        shards = zip(src_shards, tgt_shards)
    else:
        shards = zip(src_shards, agenda_shards, tgt_shards)

    for i, flat_shard in enumerate(shards):
        if not opt.agenda:
            src_shard, tgt_shard = flat_shard
            agenda_shard = None
        else:
            src_shard, agenda_shard, tgt_shard = flat_shard
        logger.info("Translating shard %d." % i)

        tag_shard = None
        if opt.constraint_file:
            tag_shard = next(tag_shards)

        translator.translate(
            src=src_shard,
            tgt=tgt_shard,
            agenda=agenda_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug,
            tag_shard=tag_shard
            )


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    
    model_path = opt.models[0]
    step = os.path.basename(model_path)[:-3].split('step_')[-1]
    temp = opt.random_sampling_temp

    if opt.extra_output_str:
        opt.extra_output_str = '_'+opt.extra_output_str

    if opt.output is None:
        output_path = '/'.join(model_path.split('/')[:-2])+'/output_%s_%s%s.encoded' % (step, temp, opt.extra_output_str)
        opt.output = output_path
    print(opt.output)

    main(opt)
