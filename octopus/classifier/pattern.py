import re
import os
import dill

from collections import Counter


class PatternClassifier:
    def __init__(self, model_path: str = None):
        self.n_str_pattern = re.compile(pattern='[\\d\\-?/_!\\.,]')
        self.doublespacing = re.compile(pattern='\\s\\s+')
        self.string_only = re.compile(pattern='[^a-z가-힣]+')

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
        score, pred = 0, 0
        sent = self.preprocess([sent])[0]
        sent = sent.replace(' ', '')
        sent = self.string_only.sub(repl='', string=sent)

        patterns = re.findall(self.pattern, string=sent)
        if patterns:
            score = sum([len(s) for s in patterns]) / len(sent)

        if score >= threshold:
            pred = 1
        return {'pred': pred, 'score': score}

    def preprocess(self, sents: list):
        sents = [self.n_str_pattern.sub(repl=' ', string=w) for w in sents]
        sents = [self.doublespacing.sub(repl=' ', string=w).strip() for w in sents]
        sents = [u.lower() for u in sents]
        return sents

    def save_model(self, save_path, save_prefix):
        path = os.path.join(save_path, save_prefix+'.model')

        outp = {'counts': self.counts, 'pattern': self.pattern}
        with open(path, 'wb') as saveFile:
            dill.dump(outp, saveFile)