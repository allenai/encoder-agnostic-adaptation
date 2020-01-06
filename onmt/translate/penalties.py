from __future__ import division
import torch


class PenaltyBuilder(object):
    """Returns the Length and Coverage Penalty function for Beam Search.

    Args:
        length_pen (str): option name of length pen
        cov_pen (str): option name of cov pen

    Attributes:
        has_cov_pen (bool): Whether coverage penalty is None (applying it
            is a no-op). Note that the converse isn't true. Setting beta
            to 0 should force coverage length to be a no-op.
        has_len_pen (bool): Whether length penalty is None (applying it
            is a no-op). Note that the converse isn't true. Setting alpha
            to 1 should force length penalty to be a no-op.
        coverage_penalty (callable[[FloatTensor, float], FloatTensor]):
            Calculates the coverage penalty.
        length_penalty (callable[[int, float], float]): Calculates
            the length penalty.
    """

    def __init__(self, cov_pen, length_pen):
        self.has_cov_pen = not self._pen_is_none(cov_pen)
        self.coverage_penalty = self._coverage_penalty(cov_pen)
        self.has_len_pen = not self._pen_is_none(length_pen)
        self.length_penalty = self._length_penalty(length_pen)

    @staticmethod
    def _pen_is_none(pen):
        return pen == "none" or pen is None

    def _coverage_penalty(self, cov_pen):
        if cov_pen == "wu":
            return self.coverage_wu
        elif cov_pen == "summary":
            return self.coverage_summary
        elif cov_pen == "agenda_tokens":
            return self.coverage_agenda_tokens_penalty
        elif cov_pen == "full_agenda_tokens":
            return self.coverage_full_agenda_tokens_penalty
        elif self._pen_is_none(cov_pen):
            return self.coverage_none
        else:
            raise NotImplementedError("No '{:s}' coverage penalty.".format(
                cov_pen))

    def _length_penalty(self, length_pen):
        if length_pen == "wu":
            return self.length_wu
        elif length_pen == "avg":
            return self.length_average
        elif self._pen_is_none(length_pen):
            return self.length_none
        else:
            raise NotImplementedError("No '{:s}' length penalty.".format(
                length_pen))

    # Below are all the different penalty terms implemented so far.
    # Subtract coverage penalty from topk log probs.
    # Divide topk log probs by length penalty.

    def coverage_wu(self, cov, beta=0., mask=None):
        """GNMT coverage re-ranking score.

        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        ``cov`` is expected to be sized ``(*, seq_len)``, where ``*`` is
        probably ``batch_size x beam_size`` but could be several
        dimensions like ``(batch_size, beam_size)``. If ``cov`` is attention,
        then the ``seq_len`` axis probably sums to (almost) 1.
        """

        penalty = -torch.min(cov, cov.clone().fill_(1.0)).log().sum(-1)
        return beta * penalty

    def coverage_summary(self, cov, beta=0., mask=None):
        """Our summary penalty."""
        penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(-1)
        penalty -= cov.size(-1)
        return beta * penalty

    def coverage_none(self, cov, beta=0., mask=None):
        """Returns zero as penalty"""
        none = torch.zeros((1,), device=cov.device,
                           dtype=torch.float)
        if cov.dim() == 3:
            none = none.unsqueeze(0)
        return none

    def coverage_agenda_tokens_penalty(self, cov, beta=0., mask=None):
        cov_penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(-1)
        cov_penalty -= cov.size(-1)
        cov_penalty = beta * cov_penalty

        missing_agenda_penalty = torch.max(1 - cov, cov.clone().fill_(0.0))
        missing_agenda_penalty = missing_agenda_penalty[:, :, -mask.size(1):] #Don't use non agenda
        missing_agenda_penalty = missing_agenda_penalty * mask #Don't use padding
        missing_agenda_penalty = missing_agenda_penalty.sum(-1)
        missing_agenda_penalty /= mask.sum(1)
        missing_agenda_penalty = beta * missing_agenda_penalty #TODO, have different HP
        return cov_penalty + missing_agenda_penalty

    def coverage_full_agenda_tokens_penalty(self, cov, beta=0., mask=None):
        #TODO untestes yet, and pretty slow
        cov_penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(-1)
        cov_penalty -= cov.size(-1)
        cov_penalty = beta * cov_penalty

        missing_agenda_penalty = torch.max(1 - cov, cov.clone().fill_(0.0))
        missing_agenda_penalty = missing_agenda_penalty[:, :, -mask.size(1):] #Don't use non agenda
        missing_agenda_penalty = missing_agenda_penalty * mask #Don't use padding
        missing_agenda_penalty /= mask.sum(1, keepdim=True)

        total_attn_penalty = torch.zeros_like(cov_penalty).squeeze(0)
        agenda_cov = cov[:, :, -mask.size(1):]
        for b in range(agenda_cov.size(1)):
            curr_item_prob = 1
            for c in range(agenda_cov.size(2)):
                if mask[b, c]:
                    curr_item_prob *= agenda_cov[0, b, c]
                else:
                    total_attn_penalty[b] += curr_item_prob
                    curr_item_prob = 1

        total_attn_penalty = beta * total_attn_penalty
        return cov_penalty + total_attn_penalty

    def length_wu(self, cur_len, alpha=0.):
        """GNMT length re-ranking score.

        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        return ((5 + cur_len) / 6.0) ** alpha

    def length_average(self, cur_len, alpha=0.):
        """Returns the current sequence length."""
        return cur_len

    def length_none(self, cur_len, alpha=0.):
        """Returns unmodified scores."""
        return 1.0
