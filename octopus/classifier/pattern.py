import re
import os
import dill

from collections import Counter
from octopus.utils.hangel import flat_hangeul


neg_pattern = re.compile(pattern='\\d{1,2}월|\\d{1,2}일')
digit_pattern = re.compile(pattern='\\d+')


class PatternClassifier:
    def __init__(self, model_path: str = None):
        self.n_str_pattern = re.compile(pattern='[\\-?/_!\\.,\\[\\]\\(\\)#\\+\\$&*~]')
        self.doublespacing = re.compile(pattern='\\s\\s+')
        self.string_only = re.compile(pattern='[^a-z가-힣\\s\\d]+')

        self.counts = ''
        self.pattern = ''

        if model_path:
            with open(model_path, 'rb') as file:
                model = dill.load(file)
            self.counts = model['counts']
            self.pattern = model['pattern']
            self.eng_set = set(re.findall(pattern='[a-zA-Z]+', string=self.pattern))

    def train(self, sentences, min_cnt: int = None):
        sentences = self.preprocess(sentences)
        toks = []
        for sent in sentences:
            # filter digit only tokens
            sents = [s for s in sent.split(' ') if len(s) != len(self._extract_digits(s))]
            sents = self._filter_negative_tokens(sents)
            toks += sents

        counts = Counter(toks)
        words = list(counts.keys())
        words = [w for w in words if len(w) >= 2]
        if min_cnt:
            words = [w for w in words if counts[w] >= min_cnt]
        words.sort(key=lambda item: (-len(item), item))

        self.counts = counts
        self.pattern = '|'.join(words)
        self.eng_set = set(re.findall(pattern='[a-zA-Z]+', string=self.pattern))

    def predict(self, sent: str, threshold=0.85):
        # pred: 1: in-domain, 0: out-domain
        score, pred, is_domain = 0, 0, False
        sent = self.preprocess([sent])[0]

        # add filtering case for the case when sentence is all english
        if self.is_all_eng(sent):
            outp = [s for s in sent.split(' ') if s in self.eng_set]
            score = len(' '.join(outp)) / len(sent)
        else:
            patterns = re.findall(self.pattern, string=sent)
            sent_flat = flat_hangeul(sent.replace(' ', ''))

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

        word_list = [s for s in word_list if len(s) != len(self._extract_digits(s))]
        word_list = self._filter_negative_tokens(word_list)

        words += [w for w in word_list if len(w) >= 2]
        words.sort(key=lambda item: (-len(item), item))

        self.pattern = '|'.join(words)

        for w in word_list:
            if len(w) >= 2:
                self.counts[w] = 10000

    @staticmethod
    def _extract_digits(token: str):
        match = digit_pattern.search(token)
        if match:
            s, e = match.span()
            return token[s:e]
        else:
            return ''

    @staticmethod
    def _filter_negative_tokens(token_list: list):
        rm_idx = []
        for i, token in enumerate(token_list):
            if neg_pattern.match(token):
                s, e = neg_pattern.match(token).span()
                if e-s == len(token):
                    rm_idx.append(i)
        if rm_idx:
            return [token_list[i] for i in range(len(token_list)) if i not in rm_idx]
        else:
            return token_list

    @staticmethod
    def is_all_eng(text: str):
        pattern = re.compile(pattern='[a-zA-Z\\s]+')
        match = pattern.search(text)
        if match:
            s, e = match.span()
            length = e - s
            if length == len(text):
                return True
        return False


