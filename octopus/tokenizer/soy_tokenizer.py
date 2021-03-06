# Improved soynlp LTokenizer for this package
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

import dill
import re
import os


class SoyTokenizer:
    def __init__(self, model_path: str = None):
        self.word_extractor = WordExtractor(min_frequency=5,
                                            min_cohesion_forward=0.05,
                                            min_right_branching_entropy=0.0)
        self.unk = 0
        self.pad = 1
        self.sos = 2
        self.eos = 3

        if model_path:
            with open(model_path, 'rb') as readFile:
                self.cohesion_score = dill.load(readFile)
        else:
            self.cohesion_score = {}
        self.tokenizer = LTokenizer(scores=self.cohesion_score)
        self.tok_to_id, self.id_to_tok = self._build_dict()

    def tokenize(self, sent: str):
        return self.tokenizer.tokenize(sent)

    def text_to_id(self, sent: str):
        toks = self.tokenize(sent)
        outp = []
        for s in toks:
            try:
                outp.append(self.tok_to_id[s])
            except KeyError:
                outp.append(self.unk)
        return outp

    def id_to_text(self, idxs: list):
        return [self.id_to_tok[i] for i in idxs]

    def train(self, sentences, add_whitespace: bool = False):
        sentences = self.preprocess(sentences)
        self.word_extractor.train(sentences)
        words = self.word_extractor.extract()
        self.cohesion_score = {word: score.cohesion_forward for word, score in words.items()}

        # add whitespace tokens
        if add_whitespace:
            whitetokens = []
            for s in sentences:
                whitetokens += s.split(' ')
            whitetokens = list(set(whitetokens))

            for t in whitetokens:
                self.cohesion_score.update({t: 1.0})

        self.tok_to_id, self.id_to_tok = self._build_dict()

    def save_model(self, model_path: str, model_prefix: str):
        with open(os.path.join(model_path, model_prefix+'.model'), 'wb') as saveFile:
            dill.dump(self.cohesion_score, saveFile)

    def _build_dict(self):
        tok_to_id = {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3}
        id_to_tok = {0: '<unk>', 1: '<pad>', 2: '<sos>', 3: '<eos>'}
        for i, key in enumerate(self.cohesion_score.keys()):
            tok_to_id[key] = i+4
            id_to_tok[i+4] = key
        return tok_to_id, id_to_tok

    def preprocess(self, sents: list):
        n_str_pattern = re.compile(pattern='[\\d\\-?/_!\\.,]')
        doublespacing = re.compile(pattern='\\s\\s+')

        sents = [n_str_pattern.sub(repl=' ', string=w) for w in sents]
        sents = [doublespacing.sub(repl=' ', string=w).strip() for w in sents]
        sents = [u.lower() for u in sents]
        return sents

    def __len__(self):
        return len(self.cohesion_score)
