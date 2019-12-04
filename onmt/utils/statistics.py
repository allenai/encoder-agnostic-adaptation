""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys

from onmt.utils.logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0,
                n_correct_of_first_4=0,
                n_non_padding_first_4=0,
                n_correct_agenda=0,
                n_non_padding_agenda=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_correct_of_first_4 = n_correct_of_first_4
        self.n_non_padding_first_4 = n_non_padding_first_4
        self.n_correct_agenda = n_correct_agenda
        self.n_non_padding_agenda = n_non_padding_agenda
        self.n_src_words = 0
        self.start_time = time.time()

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        from torch.distributed import get_rank
        from onmt.utils.distributed import all_gather_list

        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.n_correct_of_first_4 += stat.n_correct_of_first_4
        self.n_non_padding_first_4 += stat.n_non_padding_first_4
        self.n_correct_agenda += stat.n_correct_agenda
        self.n_non_padding_agenda += stat.n_non_padding_agenda

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def first_4_accuracy(self):
        """ compute accuracy for first 4 words"""
        if self.n_non_padding_first_4 == 0:
            return 0
        return 100 * (self.n_correct_of_first_4 / self.n_non_padding_first_4)

    def agenda_accuracy(self):
        """ compute accuracy for first 4 words"""
        if self.n_non_padding_agenda == 0:
            return 0
        return 100 * (self.n_correct_agenda / self.n_non_padding_agenda)

    def xent(self):
        """ compute cross entropy """
        return self.loss / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
                ("Step %s; acc: %6.2f; first_4_acc: %6.2f; agenda_acc: %6.2f; ppl: %5.2f; xent: %4.3f; " +
             "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (step_fmt,
               self.accuracy(),
               self.first_4_accuracy(),
               self.agenda_accuracy(),
               self.ppl(),
               self.xent(),
               learning_rate,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        if self.n_non_padding_first_4 > 0:
            writer.add_scalar(prefix + "/first_4_accuracy", self.first_4_accuracy(), step)
        if self.n_non_padding_agenda > 0:
            writer.add_scalar(prefix + "/agenda_accuracy", self.agenda_accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
