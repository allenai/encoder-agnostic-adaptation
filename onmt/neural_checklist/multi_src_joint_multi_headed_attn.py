""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from onmt.utils.misc import generate_relative_positions_matrix,\
                            relative_matmul


"""
This should be in multi_headed_attn.py
"""

class MultiSrcJointMultiHeadedAttention(nn.Module):
    """
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_positions=0, ctx_weight_param=False, num_src=2):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiSrcJointMultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.ctx_linear_keys = nn.ModuleList([nn.Linear(model_dim,
                                     head_count * self.dim_per_head) for _ in range(num_src)])
        self.ctx_linear_values = nn.ModuleList([nn.Linear(model_dim,
                                     head_count * self.dim_per_head) for _ in range(num_src)])
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)

        if ctx_weight_param:
            self.ctx_bias = Parameter(torch.ones(1)*-10)
        self.ctx_weight_param = ctx_weight_param

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def forward(self, self_kvq, ctx_kv, self_mask=None, ctx_mask=None,
                layer_cache=None, type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           self_kvq (FloatTensor): set of `self_len`
               key vectors ``(batch, self_len, dim)``
           ctz_kv (List[FloatTensor]): set of `ctx_len`
               value vectors ``(batch, ctx1_len, dim), (batch, ctx2_len, dim)``
           mask: binary mask indicating which keys have
               non-zero attention ``(batch, self_len, self_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, self_len, dim)``
           * one of the attention vectors ``(batch, self_len, ctx_len)``
        """

        batch_size = self_kvq.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        self_len = self_kvq.size(1)
        ctx_lens = [kv.size(1) for kv in ctx_kv]
        device = self_kvq.device

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            query, self_key, self_value = self.linear_query(self_kvq),\
                                          self.linear_keys(self_kvq),\
                                          self.linear_values(self_kvq)
            if layer_cache["self_keys"] is not None:
                self_key = torch.cat(
                    (layer_cache["self_keys"].to(device), self_key),
                    dim=1)
            if layer_cache["self_values"] is not None:
                self_value = torch.cat(
                    (layer_cache["self_values"].to(device), self_value),
                    dim=1)
            layer_cache["self_keys"] = self_key
            layer_cache["self_values"] = self_value

            if layer_cache["memory_keys"] is None:
                ctx_key = torch.cat([linear(kv) for linear, kv in zip(self.ctx_linear_keys, ctx_kv)], dim=1)
                ctx_value = torch.cat([linear(kv) for linear, kv in zip(self.ctx_linear_values, ctx_kv)], dim=1)
                layer_cache["memory_keys"] = ctx_key
                layer_cache["memory_values"] = ctx_value
            else:
                ctx_key = layer_cache["memory_keys"]
                ctx_value = layer_cache["memory_values"]
        else:
            self_key = self.linear_keys(self_kvq) # [batch, self_len, dim]
            self_value = self.linear_values(self_kvq)
            query = self.linear_query(self_kvq)
            ctx_key = torch.cat([linear(kv) for linear, kv in zip(self.ctx_linear_keys, ctx_kv)], dim=1)
            ctx_value = torch.cat([linear(kv) for linear, kv in zip(self.ctx_linear_values, ctx_kv)], dim=1)

        self_len = self_key.size(1) # Need to do this again to include the layer_cache length 

        key = torch.cat((self_key, ctx_key), dim=1)
        value = torch.cat((self_value, ctx_value), dim=1)

        key = shape(key)
        value = shape(value)

        if self.max_relative_positions > 0 and type == "self":
            raise NotImplementedError
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(device))

        query = shape(query)

        key_len = key.size(2) # self_len+ctx_len
        query_len = query.size(2) # self_len

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3)) # [batch, head, self_len, self_len+ctx_len]

        if self.ctx_weight_param:
            query_key[..., self_len:] += self.ctx_bias
        #print(query_key.mean(), query_key.std())

        if self.max_relative_positions > 0 and type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if self_mask is not None:
            self_mask = self_mask.unsqueeze(1)  # [B, 1, self_len, self_len]
            scores[:, :, :, :self_len] = scores[:, :, :, :self_len].masked_fill(self_mask, -1e18)
        if ctx_mask is not None:
            for i in range(len(ctx_mask)):
                if i == 0:
                    scores[:, :, :, self_len:self_len+ctx_lens[i]] = scores[:, :, :, self_len:self_len+ctx_lens[i]].masked_fill(ctx_mask[i].unsqueeze(1), -1e18)
                else:
                    scores[:, :, :, self_len+sum(ctx_lens[:i]):] = scores[:, :, :, self_len+sum(ctx_lens[:i]):].masked_fill(ctx_mask[i].unsqueeze(1), -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value) # [batch, head, self_len, dim]

        if self.max_relative_positions > 0 and type == "self":
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False))
        else:
            context = unshape(context_original)

        output = self.final_linear(context)

        # Return one attn (to context)
        ctx_attn_probs = attn[:, :, :, self_len:]
        ctx_attn_probs = ctx_attn_probs/(ctx_attn_probs.sum(dim=-1, keepdim=True) + 1e-20)

        top_attn = ctx_attn_probs \
            .view(batch_size, head_count,
                  query_len, -1)[:, 0, :, :] \
            .contiguous()

        return output, top_attn, attn
