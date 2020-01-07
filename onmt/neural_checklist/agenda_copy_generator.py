import torch
import torch.nn as nn

import onmt
from onmt.utils.misc import aeq
from onmt.utils.loss import LossComputeBase
from onmt.modules.copy_generator import collapse_copy_scores

FORCE_PTRS = True

class AgendaCopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.

    These networks consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx):
        super(AgendaCopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map, align=None, ptrs=None, tags=None):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        # aeq(slen, slen_) #This is not true anymore because I'm only copying from agenda part
        full_src_len, agenda_len, context_len = slen, slen_, slen - slen_

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))

        if self.training and (ptrs is not None or FORCE_PTRS):
            align_unk = align.eq(0).float().view(-1, 1)
            align_not_unk = align.ne(0).float().view(-1, 1)
            out_prob = torch.mul(prob, align_unk)
            mul_attn = torch.mul(attn[:, context_len:], align_not_unk)
            if not FORCE_PTRS:
                # This is not used as not sure how ptrs should look like
                mul_attn = torch.mul(mul_attn, ptrs.view(-1, agenda_len).float())
        else:
            out_prob = torch.mul(prob, 1 - p_copy)
            # Mask disallowed copys
            if tags is not None:
                mul_attn = torch.mul(attn[:, context_len:], tags.t())*2
            else:
                mul_attn = attn[:, context_len:]
            mul_attn = torch.mul(mul_attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, agenda_len).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1), p_copy

class CopyGeneratorWithAgendaLossCompute(LossComputeBase):
    """Copy Generator Loss Computation."""
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length, ptrs_loss=False):
        super(CopyGeneratorWithAgendaLossCompute, self).__init__(criterion, generator)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length

        self.ptrs_loss = ptrs_loss
        if ptrs_loss:
            self.switch_loss_criterion = nn.BCELoss(reduction='sum')

    def _make_shard_state(self, batch, output, range_, attns):
        """See base class for args description."""
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        ptrs = batch.ptrs[range_[0] + 1: range_[1]] if hasattr(batch, 'ptrs') else None

        ret_dict= {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]],
            "ptrs": ptrs
        }

        return ret_dict

    def _compute_loss(self, batch, output, target, copy_attn, align, ptrs):
        """Compute the loss.

        The args must match :func:`self._make_shard_state()`.

        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """

        target = target.view(-1)
        align = align.view(-1)
        src_map = batch.src_map.to(dtype=output.dtype)

        scores, p_copy = self.generator(
            self._bottle(output), self._bottle(copy_attn), src_map,
            align=align, ptrs=ptrs
        )
        loss = self.criterion(scores, align, target)

        # ptr stuff
        if self.ptrs_loss:
            # align needs to be bigger than 1 to be encouraged.
            switch_loss = self.switch_loss_criterion(p_copy, align.ne(0).float().view(-1, 1))
        else:
            switch_loss = 0

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, batch.dataset.src_vocabs)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone(), scores_data, target_data, batch, switch_loss.sum().clone(), align.ne(0).sum())

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()
        
        if self.ptrs_loss:
            loss = loss + switch_loss

        return loss, stats

    def _stats(self, loss, scores, target, batch, switch_loss, n_copied_tokens):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.view(-1).ne(self.padding_idx)
        num_correct = pred.eq(target.view(-1)).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()

        max_length = batch.tgt.size(0) - 1
        full_pred = pred.view(max_length, -1)
        n_correct_agenda, n_non_padding_agenda = 0, 0
        for b in range(batch.agenda[0].size(1)):
            items = self._find_non_padding_items(batch.agenda[0][:, b, 0])

            found_items = [self._search_for_full_item(full_pred[:, b], item) for item in items]
            count_max_1_found = [min(found, 1) for found in found_items]
            n_correct_agenda += sum(count_max_1_found)
            n_non_padding_agenda += len(items)

        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct, n_correct_agenda=n_correct_agenda, n_non_padding_agenda=n_non_padding_agenda, switch_loss=switch_loss.item(), n_copied_tokens=n_copied_tokens)

    def _find_non_padding_items(self, items):
        ret_items = []
        inside_outside_item = 'o'
        for item in items:
            if item == self.padding_idx:
                inside_outside_item = 'o'
            else:
                if inside_outside_item == 'o':
                    inside_outside_item = 'i'
                    ret_items.append([item])
                else:
                    ret_items[-1].append(item)

        return ret_items

    def _search_for_full_item(self, pred, item):
        found = 0
        starts = (pred == item[0]).nonzero().squeeze(1)
        for start in starts:
            found += 1
            pointer = start
            for x in item[1:]:
                pointer = pointer+1
                if pointer >= pred.size(0):
                    break
                if not pred[pointer] == x:
                    found -= 1
                    break

        return found
