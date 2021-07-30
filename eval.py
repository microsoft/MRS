"""Compute evaluation scores.

The input is a TSV file. Each line has five columns:
Message <tab> Reference <tab> Prediction 1 <tab> Prediction 2 <tab> Prediction 3

For japanese, use the --ja flag.
"""

from argparse import ArgumentParser
import json

from nltk.tokenize import word_tokenize
from rouge import Rouge
from rouge.rouge_score import rouge_n


def main():
    parser = ArgumentParser()
    parser.add_argument('file', help='path to prediction file (tsv)')
    parser.add_argument('--nbest', type=int, default=3, help='number of hypothesis to use')
    parser.add_argument('--max_hyp_len', type=int, default=100,
                        help='max number of words in hypothesis (default=100)')
    parser.add_argument('--min_tgt_len', type=int, default=0,
                        help='skip examples with reponses that are too long.')
    parser.add_argument('--max_tgt_len', type=int, default=100,
                        help='skip examples with reponses that are too long.')
    parser.add_argument('--ja', action='store_true', help='Use Japanese tokenizer')
    args = parser.parse_args()

    if args.ja:
        import MeCab
        wakati = MeCab.Tagger("-Owakati")
        tokenizer = lambda text: wakati.parse(text)
    else:
        tokenizer = lambda text: ' '.join(word_tokenize(text))

    refs = []
    all_hyps = []
    with open(args.file, 'r') as f:
        for line in f:
            fields = line.lower().strip().split('\t')
            ref = tokenizer(fields[1])
            if args.min_tgt_len <= len(ref.split()) <= args.max_tgt_len:
                refs.append(ref)
                if args.nbest is None:
                    hyps = fields[2:]
                else:
                    hyps = fields[2:2+args.nbest]
                all_hyps.append([tokenizer(hyp) for hyp in hyps])

    # Filter examples where the reference only contains dots
    # The rogue package breaks down on these examples
    n_empty = 0
    filtered_refs, filtered_all_hyps = [], []
    for ref, hyps in zip(refs, all_hyps):
        filtered_hyps = []
        for hyp in hyps:
            hyp_fields = [sent for sent in hyp.split('.') if len(sent) > 0]
            if len(hyp_fields) == 0:
                filtered_hyps.append('<empty>')
                n_empty += 1
            else:
                # This is necessary as ROGUE assumes punctuations are tokenized
                filtered_hyps.append(hyp)
        if len(hyps) == 0:
            filtered_hyps.append('<empty>')
            n_empty += 1

        ref_fields = [sent for sent in ref.split('.') if len(sent) > 0]
        if len(ref_fields) > 0:
            filtered_refs.append(ref)
            filtered_all_hyps.append(filtered_hyps)
    print('Find {} empty hypothesis.'.format(n_empty))
    
    Rouge.DEFAULT_METRICS.append('rouge-3')
    Rouge.AVAILABLE_METRICS['rouge-3'] = lambda hyp, ref, **k: rouge_n(hyp, ref, 3, **k)
    rogue_evaluator = Rouge()

    # For each example, select the hypothesis with the highest ROGUE score
    filtered_hyps = []
    n_empty = 0
    for ref, hyps in zip(filtered_refs, filtered_all_hyps):
        max_score = -1
        best_hyp = None
        for hyp in hyps:
            if len(hyp.split()) > args.max_hyp_len:
                hyp = ' '.join(hyp.split()[:args.max_hyp_len])
            scores = rogue_evaluator.get_scores(hyp, ref)[0]
            ensembled_score = (
                scores['rouge-1']['f'] / 6
                + scores['rouge-2']['f'] / 3
                + scores['rouge-3']['f'] / 2
            )
            if ensembled_score > max_score:
                best_hyp = hyp
                max_score = ensembled_score
        filtered_hyps.append(best_hyp)

    hyps, refs = filtered_hyps, filtered_refs

    scores = rogue_evaluator.get_scores(hyps, refs, avg=True)
    scores['n_examples'] = len(refs)
    scores['rouge-weighted'] = {}
    for field in ('f', 'r', 'p'):
        scores['rouge-weighted'][field] = (
            scores['rouge-1'][field] / 6
            + scores['rouge-2'][field] / 3
            + scores['rouge-3'][field] / 2
        )

    distinct_1, distinct_2 = set(), set()
    for hyp in hyps:
        words = hyp.split()
        distinct_1.update(words)
        distinct_2.update((words[i], words[i + 1])
                          for i in range(len(words) - 1))
    scores['distinct-1'] = len(distinct_1) / tot_len
    scores['distinct-2'] = len(distinct_2) / tot_len

    print(json.dumps(scores, indent=4))


if __name__ == '__main__':
    main()
