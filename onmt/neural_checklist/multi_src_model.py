""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn


class MultiSrcNMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoders, decoder):
        super(MultiSrcNMTModel, self).__init__()
        self.encoders = encoders
        self.decoder = decoder

    def encode(self, srcs, length_tuples):
        enc_state, memory_bank, lengths_out = [], [], []

        for src, encoder, lengths in zip(srcs, self.encoders, length_tuples):
            outputs = encoder(src, lengths)
            enc_state.append(outputs[0])
            memory_bank.append(outputs[1])
            lengths_out.append(outputs[2])

        return enc_state, memory_bank, lengths_out

    def maybe_update_check_vec(self, log_probs):
        """
        Args: log_probs: (Tensor): tgt_len * batch * vocab_size.
            scores are before softmax
        """
        self.decoder.maybe_update_check_vec(log_probs)

    def forward(self, srcs, tgt, length_tuples, bptt=False, **kwargs):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            length_tuples(LongTensor): The src length_tuples, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_state, memory_bank, lengths_out = self.encode(srcs, length_tuples)
        if bptt is False:
            self.decoder.init_state(srcs, memory_bank, enc_state)

        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths_out, **kwargs)

        return dec_out, attns
