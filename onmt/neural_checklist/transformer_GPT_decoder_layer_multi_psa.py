import torch
from torch import nn

from .multi_src_joint_multi_headed_attn import MultiSrcJointMultiHeadedAttention

from onmt.modules.gpt_mlp import MLP

class TransformerGPTDecoderLayerMultiPSA(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attn_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 ctx_weight_param=False, num_src=2):
        super(TransformerGPTDecoderLayerMultiPSA, self).__init__()
        
        # This is called self for easier loading of gpt params
        self.self_attn = MultiSrcJointMultiHeadedAttention(
            heads, d_model, dropout=attn_dropout,
            max_relative_positions=max_relative_positions,
            ctx_weight_param=ctx_weight_param, num_src=num_src)

        self.feed_forward = MLP(d_model, d_model*4, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.drop = nn.Dropout(dropout)
        #self.ctx_weight = Parameter(torch.zeros(1))

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None, evaluate_attns=False):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``

        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)

        query, attn, all_attn_probs = self.self_attn(input_norm, memory_bank, self_mask=dec_mask,
                                                     ctx_mask=src_pad_mask, layer_cache=layer_cache)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)

        output = self.feed_forward(query_norm)
        output = output + query

        if evaluate_attns:
            return output, attn, all_attn_probs
        else:
            return output, attn
