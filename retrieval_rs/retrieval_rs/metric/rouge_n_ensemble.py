from retrieval_rs.constants.weights import SCORE_ERROR
from retrieval_rs.metric import rouge
import sys
import os

SR_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, SR_DIR)


class Sentence_ROUGE_n():
    def __init__(self, metric_name, stats_name, suffix=""):
        self.metric_name = metric_name
        self.stats_name = stats_name
        self.name = "sROUGE_" + metric_name + "_" + stats_name
        self.name += suffix
        self.store = list()

    def compute(self, rouge_n_for_one_sample):
        '''
        get the rouge_[1,2,3,L]_[f,p,r] score for one row of sample from the pre-computed dictionary,
        and the score is the maximum of the top3 responses
        :param rouge_n_for_one_sample: a list of metric dictionary for the top3 responses
        :return: the concrete rouge_[1,2,3,L]_[f,p,r] score
        '''
        # score should be max of top3
        score = SCORE_ERROR
        for i in range(len(rouge_n_for_one_sample)):
            if rouge_n_for_one_sample[i] != SCORE_ERROR:
                score = max(score, rouge_n_for_one_sample[i]['rouge-'+self.metric_name][self.stats_name])
        return score

    def compute_add(self, rouge_n_for_one_sample):
        """
        store the score of each sample for each metric, which is used to calculate the average and pvalue
        :param rouge_n_for_one_sample:  a list of metric dictionary for the top3 responses
        :return: the concrete rouge_[1,2,3,L]_[f,p,r] score
        """
        score = self.compute(rouge_n_for_one_sample)
        # do not aggregate if score is ERROR
        if not score == SCORE_ERROR:
            self.store.append(score)
        return score

    def aggregate(self):
        """
        calculate the average of all samples
        :return: average score
        """
        return sum(self.store) / len(self.store) if len(self.store) > 0 else 0

    def get_store(self):
        return self.store


class Sentence_ROUGE_ensemble_f():
    def __init__(self, ensemble_name, suffix=""):
        """
        the ensemble rouge, can be either rouge_uniform_f or rouge_weight_f
        :param ensemble_name: "uniform"/"weight"
        """
        self.ensemble_name = ensemble_name
        self.name = "Sentence_ROUGE_" + ensemble_name + "_f"
        self.name += suffix
        self.rouge = rouge.Rouge()
        self.store = list()

    def compute_add(self, one_sample_scores, rouge1f_idx, rouge2f_idx, rouge3f_idx, unigram_weight, bigram_weight, trigram_weight):
        if one_sample_scores[rouge1f_idx] == SCORE_ERROR or one_sample_scores[rouge2f_idx] == SCORE_ERROR \
                or one_sample_scores[rouge3f_idx] == SCORE_ERROR:
            return SCORE_ERROR
        if self.ensemble_name == "uniform":
            score = (one_sample_scores[rouge1f_idx] + one_sample_scores[rouge2f_idx] + one_sample_scores[rouge3f_idx])/3
        else:
            score = one_sample_scores[rouge1f_idx] * unigram_weight + one_sample_scores[rouge2f_idx] * bigram_weight + \
                one_sample_scores[rouge3f_idx] * trigram_weight
        self.store.append(score)
        return score

    def aggregate(self):
        return sum(self.store) / len(self.store) if len(self.store) > 0 else 0

    def get_store(self):
        return self.store

class Avg_Response_Length():
    def __init__(self, suffix=""):
        """
        the average length of model predicted responses
        :param "suffix"="final"/"undeduped"
        """
        self.name = "avg_response_length" + suffix
        self.store = list()

    def compute_add(self, model_responses):
        length = 0
        for i in range(len(model_responses)):
            length += len(model_responses[i])
            score = length/len(model_responses)
        self.store.append(score)
        return score

    def aggregate(self):
        return sum(self.store) / len(self.store) if len(self.store) > 0 else 0
