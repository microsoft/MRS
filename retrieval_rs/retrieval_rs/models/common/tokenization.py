"""
File contains tokenization classes
-   Vocab Class: for handling vocabulary and converting tokenized string to indices
-   Tokenizer Class: A simple unicode based tokenizer (Copyright 2018 The Google AI Language Team Authors from:
    https://github.com/google-research/bert/blob/master/tokenization.py). The tokenizer has been extended to support
    emojis and emoticons as a single token.
-   WordpieceTokenizer Class: wordpiece tokenizer (Copyright 2018 The Google AI Language Team Authors from:
    https://github.com/google-research/bert/blob/master/tokenization.py).
-   FullTokenizer Class: end to end tokenizer (Copyright 2018 The Google AI Language Team Authors from:
    https://github.com/google-research/bert/blob/master/tokenization.py).
-   SentencepieceTokenizer Class: end to end sentencepiece tokenizer for text (Copyright 2016 Google Inc. from: https://github.com/google/sentencepiece).
@authors: Pankaj Gulhane, Budhaditya Deb, Guilherme Ilunga, Shashank Jain, Lili Zhou
"""

import os
import re
import six
import unicodedata
import json
import shutil
import argparse
try:
    import sentencepiece as spm
except Exception as ex:
    print("SystemLog: spm (sentencepiece) not found while importing in utils %s" % ex)
from transformers import XLMRobertaTokenizer


# TODO: There should be a separate vocab module for this Vocab class
class Vocab(object):
    def __init__(self, data_reader, vocab_path, record_key, min_count=1, initial_vocab=None,
                 _PAD="<pad>", _UNK="<unk>", _EOS="<eos>", verbose=False):
        self.data_reader = data_reader
        self.key = record_key
        self.vocab_path = vocab_path
        self.min_count = min_count
        self.verbose = verbose

        # FIXME: This entire logic of fixing IDs for special tokens combined with existing vocab loading
        # is buggy; when we load the existing vocab, we redefine these IDs and thus the same ID is used for
        # multiple tokens

        if initial_vocab is None:
            self.initial_vocab = ["<pad>", "<unk>"]
        else:
            self.initial_vocab = initial_vocab
        self.TOKEN_PAD_ID = _PAD
        self.TOKEN_UNK_ID = _UNK

        self.vocab = {}
        self.rev_vocab = []

    def convert_tokens_to_ids(self, tokens, removeUnkTokens=False):
        out = []
        if removeUnkTokens:
            out = [self.vocab[w] for w in tokens if w in self.vocab]
        else:
            out = [self.vocab.get(w, self.TOKEN_UNK_ID) for w in tokens]
        return out

    def convert_ids_to_tokens(self, token_ids):
        tokens = token_ids
        if self.TOKEN_EOS_ID in token_ids:
            tokens = token_ids[:token_ids.index(self.TOKEN_EOS_ID)]
        out = [self.rev_vocab[i] for i in tokens]
        return out

    def load_vocab(self):
        if self.verbose:
            print("SystemLog: Loading vocabulary from {}".format(self.vocab_path))
        self.rev_vocab = []
        with open(self.vocab_path, mode="r", encoding="utf-8") as f:
            self.rev_vocab.extend(f.readlines())
            self.rev_vocab = [line.strip() for line in self.rev_vocab]
            self.vocab = dict([(x, y) for (y, x) in enumerate(self.rev_vocab)])

    def get_vocab_size(self):
        return len(self.vocab)

    def get_unk_id(self):
        return self.TOKEN_UNK_ID

    def get_pad_id(self):
        return self.TOKEN_PAD_ID

    def init_vocab(self, vocab_size):
        if not os.path.isfile(self.vocab_path):
            raise ValueError("SystemLog: Implement Vocab creation")
            _init_vocab(self.vocab_path, self.data_reader, self.key,
                        vocab_size, self.initial_vocab, self.min_count, -1)
        self.load_vocab()

        # Set below to the correct vocab id's
        if self.TOKEN_PAD_ID in self.vocab:
            self.TOKEN_PAD_ID = self.vocab[self.TOKEN_PAD_ID]

        if self.TOKEN_UNK_ID in self.vocab:
            self.TOKEN_UNK_ID = self.vocab[self.TOKEN_UNK_ID]


class Tokenizer(object):
    # Pipeline: unicode/cleanup --> lowercase --> normalize digits --> strip accents --> whitespace --> add begin/end tokens
    def __init__(self, vocab_path=None, use_sentence_start_end=True, make_lowercase=True, strip_accents=True, emoticons_path=None, emojis_path=None, remove_tee_tags=False, tokenizer_type=None, remove_unk_tokens=False):
        if emoticons_path is None:
            self._DIGIT_RE = re.compile(r"\d+")
        else:
            # Avoid replacing numbers when they are part of an emoticon
            self._DIGIT_RE = re.compile(r"(?<!([<=:]))(?<!(</))(?<!(:-))(?<!(<\\))\d(?!\))(?!-\))")
        self.use_sentence_start_end = use_sentence_start_end
        self.make_lowercase = make_lowercase
        self.strip_accents = strip_accents
        self.emoticon_regex = self.emoji_set = None
        self.remove_tee_tags = remove_tee_tags
        self.remove_unk_tokens = remove_unk_tokens
        self.tokenizer_type = tokenizer_type

        # Remove O.com special tokens introduced in TEE pre-processing
        self.BOS_RE = re.compile(r"#bos#|#eos#", re.IGNORECASE)
        self.entities_regex_str = r"#bos#|#eos#|#name#|#date#|#time#|#datetime#|#url#|#fulladdress#|#phonenumber#|#email#"
        # Converting TEE tags to unused tokens in the BERT vocab
        self.tee_vocab_map = {k: '[unused' + str(i+1) + ']' for i, k in enumerate(self.entities_regex_str.split("|"))}

        #unused item should be never splitted when using BERT vocab, otherwise the entity taggers should not be splitted.
        if self.tokenizer_type == "wordpiece":
            self.never_split = list(self.tee_vocab_map.values())
        else:
            self.never_split = list(self.tee_vocab_map.keys())

        if emoticons_path is not None:
            if not os.path.isfile(emoticons_path):
                raise ValueError("SystemLog: Emoticons file is not in the provided path (path: {})".format(emoticons_path))
            with open(emoticons_path, "r", encoding="utf-8") as f:
                emoticons = json.load(f)
            self.emoticon_regex = re.compile(emoticons["all"])

        if emojis_path is not None:
            if not os.path.isfile(emojis_path):
                raise ValueError("SystemLog: Emojis file is not in the provided path (path: {})".format(emojis_path))
            with open(emojis_path, "r", encoding="utf-8") as f:
                emojis = json.load(f)
            self.emoji_set = set(v["UNICODE_STR"] for v in emojis.values())

        self.vocab = Vocab(None, vocab_path, None, min_count=-1, initial_vocab=None, _PAD="<pad>", _UNK="<unk>")
        self.vocab.init_vocab(-1)

    def convert_tokens_to_ids(self, tokens):
        return self.vocab.convert_tokens_to_ids(tokens, self.remove_unk_tokens)

    def convert_ids_to_tokens(self, ids):
        return self.vocab.convert_ids_to_tokens(ids)

    def get_vocab_size(self):
        return self.vocab.get_vocab_size()

    def split_emojis(self, text: str) -> str:
        """
        When an emoji is found, split it from text or other emojis with a space character, except the zero-width joiner
        (\u200d) and var selector (\ufe0f).

        :param text: The text which may contain emojis to split.
        :return: The processed text, where emojis are separated by a space character.
        """
        processed_text = ""

        for char in text:
            if char in self.emoji_set:
                # if last char is not whitespace or special emoji joiner characters, must add a space before the emoji
                if len(processed_text) > 0 and re.match(r"\S", processed_text[-1]) and \
                        processed_text[-1] not in [u"\u200d", u"\ufe0f"]:
                    processed_text += " "
                    processed_text += char
                else:
                    processed_text += char
            else:
                # if last char is an emoji and current is not a special emoji joiner character, must add a space
                if len(processed_text) > 0 and char not in [u"\u200d", u"\ufe0f"] and \
                        processed_text[-1] in self.emoji_set:
                    processed_text += " "
                    processed_text += char
                else:
                    processed_text += char

        return processed_text

    def split_emoticons(self, text: str) -> list:
        """
        This function processes a string of text into a list of characters. If the text contains emoticons, then they
        are returned as a single element in the list.

        :param text: The text which may contain emojis.
        :return: The list of characters in the text, with emoticons being considered a single character.
        """
        if re.search(self.emoticon_regex, text):
            tokens = [w for w in re.split(self.emoticon_regex, text) if w is not None and len(w) > 0]
            chars = []
            for token in tokens:
                if re.match(self.emoticon_regex, token):
                    chars.append(token)
                elif len(token) > 1:
                    chars.extend(token)
        else:
            chars = list(text)

        return chars

    def tokenize(self, text):
        """Pre-tokenizer used in wordpiece, assuming utf-8 input."""
        # TODO: entities_regex_str should be a seperate token, but is that even needed?
        # TODO: remove_boseos is also something not required I believe.
        text = self.convert_to_unicode(text)
        text = self.clean_text(text)
        text = re.sub(self._DIGIT_RE, "0", text)  # remove numbers

        # Remove O.com special tokens introduced in TEE pre-processing
        if self.remove_tee_tags:
            text = re.sub(self.BOS_RE, " ", text)
            text = re.sub(re.compile(r'('+self.entities_regex_str+r')', re.IGNORECASE), r' \1 ', text)

        if self.emoji_set is not None:
            text = self.split_emojis(text)

        orig_tokens = self.whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.make_lowercase:
                token = token.lower()
            if self.remove_tee_tags and self.tokenizer_type == "wordpiece":
                token = self.replace_tee_tag_with_unused_vocab(token)
            if self.strip_accents:
                token = self.run_strip_accents(token)
            split_tokens.extend(self.run_split_on_punc(token))

        tokens = self.whitespace_tokenize(" ".join(split_tokens))

        if self.use_sentence_start_end:
            tokens = ["<s>"] + tokens + ["</s>"]
        return tokens

    def replace_tee_tag_with_unused_vocab(self, token):
        """ Replace TEE tags with unused tokens in BERT vocab """
        if token.lower() in self.tee_vocab_map:
            return self.tee_vocab_map[token.lower()]
        return token

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("SystemLog: Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("SystemLog: Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("SystemLog: Not running on Python2 or Python 3?")

    def clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self.is_control(char):
                continue
            if self.is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def run_split_on_punc(self, text):
        """Splits punctuation on a piece of text.
        If punctuation is part of an emoticon, then does not split."""
        if self.remove_tee_tags and text in self.never_split:
            return [text]
        if self.emoticon_regex is not None:
            chars = self.split_emoticons(text)
        else:
            chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if (self.emoticon_regex is not None and re.match(self.emoticon_regex, char)) or self.is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def is_punctuation(self, char):
        """Checks whether `chars` is a punctuation character."""
        cp = ord(char)
        # We treat all non-letter/number ASCII as punctuation.
        # Characters such as "^", "$", and "`" are not in the Unicode
        # Punctuation class but we treat them as punctuation anyways, for
        # consistency.
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
                (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def is_control(self, char):
        """Checks whether `chars` is a control character.
         The zero-width joiner (\u200d) and var selector (\ufe0f) unicode characters are not considered control
         characters"""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C") and char not in [u"\u200d", u"\ufe0f"]:
            return True
        return False

    def is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

# TODO: These Tokenizers should be in their own separate modules


class FullTokenizer(Tokenizer):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_path, use_sentence_start_end=False, make_lowercase=True, strip_accents=True, emoticons_path=None, emojis_path=None, remove_tee_tags=False, remove_unk_tokens=False):
        self.remove_unk_tokens = remove_unk_tokens
        self.basic_tokenizer = Tokenizer(vocab_path=vocab_path, use_sentence_start_end=False, make_lowercase=make_lowercase, strip_accents=strip_accents,
                                         emoticons_path=emoticons_path, emojis_path=emojis_path, remove_tee_tags=remove_tee_tags, tokenizer_type="wordpiece", remove_unk_tokens=remove_unk_tokens)  # bert BasicTokenizer
        self.vocab = Vocab(None, vocab_path, None, min_count=-1, initial_vocab=None, _PAD="[PAD]", _UNK="[UNK]", _EOS="[SEP]")
        self.vocab.init_vocab(-1)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab.vocab, remove_unk_tokens=remove_unk_tokens)
        self.use_sentence_start_end = use_sentence_start_end

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        if self.use_sentence_start_end:
            split_tokens = ["[CLS]"] + split_tokens + ["[SEP]"]

        return split_tokens


# TODO: These Tokenizers should be in their own separate modules
class WordpieceTokenizer(Tokenizer):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200, remove_unk_tokens=False):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.remove_unk_tokens = remove_unk_tokens

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]
        Args:
            text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.
        Returns:
            A list of wordpiece tokens.
        """

        text = self.convert_to_unicode(text)
        output_tokens = []
        for token in self.whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class XLMRobertaTokenizerWrapper(XLMRobertaTokenizer):
    def __init__(self, vocab_path, use_sentence_start_end=False, remove_tee_tags=False):
        super().__init__(vocab_path)

        self.use_sentence_start_end = use_sentence_start_end
        print("SystemLog: start_token={}\tend_token={}".format(self.cls_token, self.sep_token))
        if self.use_sentence_start_end:
            assert (self.cls_token is not None)
            assert (self.sep_token is not None)

        self.pad_id = 1

        self.remove_tee_tags = remove_tee_tags
        # Remove O.com special tokens introduced in TEE pre-processing
        self.BOS_RE = re.compile(r"#bos#|#eos#", re.IGNORECASE)
        self.entities_regex_str = r"#bos#|#eos#|#name#|#date#|#time#|#datetime#|#url#|#fulladdress#|#phonenumber#|#email#"

    def tokenize(self, text):
        # Remove O.com special tokens introduced in TEE pre-processing
        if self.remove_tee_tags:
            text = re.sub(self.BOS_RE, " ", text)
            text = re.sub(re.compile(r'(' + self.entities_regex_str + r')', re.IGNORECASE), r' \1 ', text)

        tokens = self._tokenize(text)
        if self.use_sentence_start_end:
            tokens = [self.cls_token] + tokens + [self.sep_token]
        return tokens

    def convert_tokens_to_ids(self, tokens):
        output = []
        for token in tokens:
            token_id = self._convert_token_to_id(token)
            if isinstance(token_id, int):
                # SentencePiece sometimes return a list of ids for strange symbols, which we ignore.
                output.append(token_id)
        return output

    def get_vocab_size(self):
        return self.vocab_size


# TODO: These Tokenizers should be in their own separate modules
class SentencepieceTokenizer(object):
    """Runs Sentencepiece tokenziation."""

    def __init__(self, vocab_train_type=None, train_data=None, vocab_path=None, vocab_size=None):
        # self.model_prefix = os.path.join(vocab_path,model_prefix)
        self.vocab_path = vocab_path
        self.model_path = self.vocab_path + '.model'
        self.vocab_size = vocab_size
        self.vocab_train_type = vocab_train_type
        self.sp = spm.SentencePieceProcessor()
        self.train_data = train_data

    def load_model(self):
        self.sp.load(self.model_path)

    def train(self):
        train_params = "--input=" + self.train_data + " --model_prefix=" + self.vocab_path + " --vocab_size=" + self.vocab_size + " --model_type=" + self.vocab_train_type
        spm.SentencePieceTrainer.train(train_params)
        print("SystemLog: sentencepiece vocab generate succsss!")

    def tokenize(self, text):
        tokens = self.sp.encode_as_pieces(text)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        output = []
        for token in tokens:
            output.append(self.sp.piece_to_id(token))
        return output

    def convert_ids_to_tokens(self, ids):
        output = []
        for id in ids:
            output.append(self.sp.id_to_piece(id))
        return output

    def get_vocab_size(self):
        return self.sp.get_piece_size()


