"""Convert format of MRS dataset."""

from argparse import ArgumentParser
import os
from glob import glob


def main():
    parser = ArgumentParser()
    parser.add_argument('input', help='original data path')
    parser.add_argument('output', help='processed data path')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    for lang in ['en', 'es', 'de', 'pt', 'fr', 'ja', 'sv', 'it', 'nl', 'ru']:
        for split in ['train', 'valid', 'test']:
            split_name = 'dev' if split == 'valid' else split
            src_path = '{}/reddit.{}.src.{}'.format(
                args.output, lang, split_name
            )
            tgt_path = '{}/reddit.{}.tgt.{}'.format(
                args.output, lang, split_name
            )
            with open(src_path, 'w') as src_file, \
                    open(tgt_path, 'w') as tgt_file:
                for filename in glob('{}/{}/{}/*.tsv*'
                                     .format(args.input, split, lang)):
                    with open(filename, 'r') as f:
                        for line in f:
                            src, tgt, _, _ = line.split('\t')
                            src = src.strip()
                            tgt = tgt.strip()
                            if len(src) > 0 and len(tgt) > 0:
                                print(src, file=src_file)
                                print(tgt, file=tgt_file)


if __name__ == '__main__':
    main()
