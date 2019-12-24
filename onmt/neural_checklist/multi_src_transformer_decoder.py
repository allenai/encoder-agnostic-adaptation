import torch
from torch import nn

from .transformer_GPT_decoder_layer_multi_psa import TransformerGPTDecoderLayerMultiPSA

from onmt.decoders.transformer import TransformerDecoder
from onmt.modules.average_attn import AverageAttention

class MultiSrcTransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attn_dropout, embeddings,
                 max_relative_positions, use_GPT_version_psa, use_GPT_version_multi_psa,
                 use_GPT_version_unconditional, use_GPT_version_ctxattn,
                 ctx_weight_param, num_src):
        super(MultiSrcTransformerDecoder, self).__init__()

        self.embeddings = embeddings

        # Decoder State
        self.state = {}
        
        kwargs = {}
        assert use_GPT_version_multi_psa
        layer_cls = TransformerGPTDecoderLayerMultiPSA
        kwargs['ctx_weight_param'] = ctx_weight_param
        kwargs['num_src'] = num_src

        self.transformer_layers = nn.ModuleList(
            [layer_cls(d_model, heads, d_ff, dropout, attn_dropout,
             self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             **kwargs)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.dec_heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout,
            opt.attn_dropout if hasattr(opt, 'attn_dropout') else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.use_GPT_version_psa,
            opt.use_GPT_version_multi_psa,
            opt.use_GPT_version_unconditional,
            opt.use_GPT_version_ctxattn,
            opt.ctx_weight_param,
            opt.num_src)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)
        
        if self.state["src"] is not None:
            for i in range(len(self.state["src"])):
                self.state["src"][i] = fn(self.state["src"][i], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        for i in range(len(self.state["src"])):
            self.state["src"][i] = self.state["src"][i].detach() if self.state["src"][i] is not None else None

    def forward(self, tgt, memory_banks, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_banks)

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        pad_idx = self.embeddings.word_padding_idx
        
        srcs = self.state["src"]
        if srcs is not None:
            src_pad_mask = []
            for src in srcs:
                src_words = src[:, :, 0].transpose(0, 1)
                src_pad_mask.append(src_words.data.eq(pad_idx).unsqueeze(1))  # [B, 1, T_src]
        else:
            src_pad_mask = None

        src_memory_bank = []
        for memory_bank in memory_banks:
            src_memory_bank.append(memory_bank.transpose(0, 1).contiguous())

        tgt_words = tgt[:, :, 0].transpose(0, 1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        save_all_attns = kwargs.get('evaluate_attns', False)
        all_attns_full = []

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            if save_all_attns:
                output, attn, all_attns = layer(
                    output,
                    src_memory_bank,
                    src_pad_mask,
                    tgt_pad_mask,
                    layer_cache=layer_cache,
                    step=step,
                    evaluate_attns=True)
                all_attns_full.append(all_attns)
            else:
                output, attn = layer(
                    output,
                    src_memory_bank,
                    src_pad_mask,
                    tgt_pad_mask,
                    layer_cache=layer_cache,
                    step=step)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if save_all_attns:
            attns['full_all_layers'] = all_attns_full

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                batch_size = memory_bank[0].size(1)
                depth = memory_bank[0].size(-1)
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth))
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache
