# -*- coding: utf-8 -*-
from __future__ import absolute_import
from retrieval_rs.metric import rouge_score as rouge_score
import sys
import os
SR_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, SR_DIR)


class Rouge:
    DEFAULT_METRICS = ["rouge-1", "rouge-2", "rouge-3", "rouge-L"]
    AVAILABLE_METRICS = {
        "rouge-1": lambda hyp, ref: rouge_score.rouge_n(hyp, ref, 1),
        "rouge-2": lambda hyp, ref: rouge_score.rouge_n(hyp, ref, 2),
        "rouge-3": lambda hyp, ref: rouge_score.rouge_n(hyp, ref, 3),
        "rouge-L": lambda hyp, ref:
            rouge_score.rouge_l_summary_level(hyp, ref),
    }
    DEFAULT_STATS = ["f", "p", "r"]
    AVAILABLE_STATS = ["f", "p", "r"]

    def __init__(self, metrics=None, stats=None):
        if metrics is not None:
            self.metrics = [m.lower() for m in metrics]

            for m in self.metrics:
                if m not in Rouge.AVAILABLE_METRICS:
                    raise ValueError("Unknown metric '%s'" % m)
        else:
            self.metrics = Rouge.DEFAULT_METRICS

        if stats is not None:
            self.stats = [s.lower() for s in stats]

            for s in self.stats:
                if s not in Rouge.AVAILABLE_STATS:
                    raise ValueError("Unknown stat '%s'" % s)
        else:
            self.stats = Rouge.DEFAULT_STATS

    def get_scores(self, hyps, refs, avg=False, ignore_empty=False):
        """
        calculate the rouge score of each pair of hyps and refs
        :param hyps: a raw text of predicted response
        :param refs: a raw text of golden response
        :param avg: average
        :param ignore_empty: Filter out hyps of 0 length
        :return: rouge score
        """
        if isinstance(hyps, str):
            hyps, refs = [hyps], [refs]

        if ignore_empty:
            # Filter out hyps of 0 length
            hyps_and_refs = zip(hyps, refs)
            hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
            hyps, refs = zip(*hyps_and_refs)

        assert(type(hyps) == type(refs))
        assert(len(hyps) == len(refs))

        if not avg:
            return self._get_scores(hyps, refs)
        return self._get_avg_scores(hyps, refs)

    def _get_scores(self, hyps, refs):
        scores = []
        for hyp, ref in zip(hyps, refs):
            sen_score = {}

            # MZ: modified to handle sentences that only have dots
            hyp_sents = [sent for sent in hyp.split('.') if len(sent) > 0]
            if len(hyp_sents) > 0:
                hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            ref_sents = [sent for sent in ref.split('.') if len(sent) > 0]
            if len(ref_sents) > 0:
                ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]
            #hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            #ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]

            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(hyp, ref)
                sen_score[m] = {s: sc[s] for s in self.stats}
            scores.append(sen_score)
        return scores

    def _get_avg_scores(self, hyps, refs):
        scores = {m: {s: 0 for s in self.stats} for m in self.metrics}

        count = 0
        for (hyp, ref) in zip(hyps, refs):
            # MZ: modified to handle sentences that only have dots
            hyp_sents = [sent for sent in hyp.split('.') if len(sent) > 0]
            if len(hyp_sents) > 0:
                hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            ref_sents = [sent for sent in ref.split('.') if len(sent) > 0]
            if len(ref_sents) > 0:
                ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]
            #hyp = [" ".join(_.split()) for _ in hyp.split(".") if len(_) > 0]
            #ref = [" ".join(_.split()) for _ in ref.split(".") if len(_) > 0]

            for m in self.metrics:
                fn = Rouge.AVAILABLE_METRICS[m]
                sc = fn(hyp, ref)
                scores[m] = {s: scores[m][s] + sc[s] for s in self.stats}
            count += 1
        scores = {m: {s: scores[m][s] / count for s in scores[m]}
                  for m in scores}
        return scores
