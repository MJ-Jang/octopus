from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline

import os

path = 'data/training'

filelist = [s for s in os.listdir(path) if s.endswith('tsv')]
except_intent = 'All_Gibberish_filter.tsv'

utterances = []
for f in filelist:
    if f == except_intent:
        pass
    else:
        with open(path + '/' + f, 'r', encoding='utf-8') as file:
            data_ = file.readlines()
        data_ = [s.strip() for s in data_]
        data_ = [s.split('\t')[1] for s in data_]
        utterances += data_


vectorizer = TfidfVectorizer()
train_vec = vectorizer.fit_transform(utterances)

model = OneClassSVM(gamma='auto')
model.fit(train_vec)


clf = make_pipeline(vectorizer, OneClassSVM(gamma='auto'))
clf.fit(utterances)
clf.predict(['하하하하하하하'])