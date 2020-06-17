import re
import os
import dill

from collections import Counter
from octopus.utils.hangel import flat_hangeul


class PatternClassifier:
    def __init__(self, model_path: str = None):
        self.n_str_pattern = re.compile(pattern='[\\d\\-?/_!\\.,\\[\\]\\(\\)#\\+\\$&*~]')
        self.doublespacing = re.compile(pattern='\\s\\s+')
        self.string_only = re.compile(pattern='[^a-z가-힣]+')

        self.counts = ''
        self.pattern = ''

        if model_path:
            with open(model_path, 'rb') as file:
                model = dill.load(file)
            self.counts = model['counts']
            self.pattern = model['pattern']

    def train(self, sentences):
        sentences = self.preprocess(sentences)
        toks = []
        for s in sentences:
            toks += s.split(' ')

        counts = Counter(toks)
        words = list(counts.keys())
        words = [w for w in words if len(w) >= 2]
        words.sort(key=lambda item: (-len(item), item))

        self.counts = counts
        self.pattern = '|'.join(words)

    def predict(self, sent: str, threshold=0.85):
        # pred: 1: in-domain, 0: out-domain
        score, pred, is_domain = 0, 0, False
        sent = self.preprocess([sent])[0]
        sent = sent.replace(' ', '')
        patterns = re.findall(self.pattern, string=sent)
        sent_flat = flat_hangeul(sent)

        if patterns:
            score = sum([len(flat_hangeul(s)) for s in patterns]) / len(sent_flat)

        if score >= threshold:
            pred = 1
            is_domain = True
        return {'pred': pred, 'score': score, 'is_domain': is_domain}

    def preprocess(self, sents: list):
        sents = [self.n_str_pattern.sub(repl=' ', string=w) for w in sents]
        sents = [self.doublespacing.sub(repl=' ', string=w).strip() for w in sents]
        sents = [u.lower() for u in sents]
        sents = [self.string_only.sub(repl='', string=u) for u in sents]
        return sents

    def save_model(self, save_path, save_prefix):
        path = os.path.join(save_path, save_prefix+'.model')

        outp = {'counts': self.counts, 'pattern': self.pattern}
        with open(path, 'wb') as saveFile:
            dill.dump(outp, saveFile)

    def add_words(self, word_list: list):
        # update words manually
        words = list(self.counts.keys())
        words = [w for w in words if len(w) >= 2]
        words += [w for w in word_list if len(w) >= 2]
        words.sort(key=lambda item: (-len(item), item))

        self.pattern = '|'.join(words)

        for w in word_list:
            if len(w) >= 2:
                self.counts[w] = 10000 

    def _automatic_threshold(self, in_domain: list, out_domain: list):
        if self.counts and self.pattern:
            id_domain_s = [self.predict(s)['socre'] for s in in_domain]
            out_domain_s = [self.predict(s)['socre'] for s in out_domain]
        else:
            raise ValueError("No model for evaluation exists")
