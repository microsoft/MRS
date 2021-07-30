"""Convert the output format of fairseq.generate to the input format of the evaluation script."""

from argparse import ArgumentParser
from collections import defaultdict


def main():
    parser = ArgumentParser()
    parser.add_argument('src', help='path to source')
    parser.add_argument('tgt', help='path to target')
    parser.add_argument('file', help='path to fairseq generate output')
    args = parser.parse_args()
    with open(args.src, 'r') as f:
        src = [line.strip() for line in f]
    with open(args.tgt, 'r') as f:
        tgt = [line.strip() for line in f]
    hyp = defaultdict(list)
    with open(args.file, 'r') as f:
        for line in f:
            if line.startswith('H-'):
                idx, _, text = line.split('\t')
                hyp[int(idx[2:])].append(text.strip())
    for i, k in enumerate(sorted(hyp.keys())):
        fields = [src[i], tgt[i]] + hyp[k]
        print('\t'.join(fields))


if __name__ == '__main__':
    main()
