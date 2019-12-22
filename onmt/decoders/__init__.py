"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder
from onmt.decoders.rnn_uncond import RNNUncondDecoder
from onmt.neural_checklist.multi_src_transformer_decoder import MultiSrcTransformerDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder,
           "rnn_uncond": RNNUncondDecoder,
           "multi_src_transformer": MultiSrcTransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec",
           "RNNUncondDecoder", "MultiSrcTransformerDecoder"]
