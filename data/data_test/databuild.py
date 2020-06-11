from rasalt.client.api_admin import get_chitchat_utterances
chitchats = get_chitchat_utterances()
sents = [s['utterance'] for s in chitchats]

with open('data/data_test/chitchat.txt', 'w', encoding='utf-8') as file:
    for line in sents:
        file.write(line + '\n')

with open('./data/data_test/nlu_goldenset.md', 'r', encoding='utf-8') as file:
    indomain = file.readlines()
indomain = [s.strip() for s in indomain]

utter = []
for s in indomain:
    if s.startswith('- '):
        utter.append(s.replace('- ', '').strip())
    elif s.startswith('## syno'):
        break

import re
pattern1 = re.compile('\([a-zA-Z_\-\d]+\)')
pattern2 = re.compile('[\[\]]+')

for i, u in enumerate(utter):
    u_ = pattern1.sub('', u)
    u_ = pattern2.sub('', u_)
    utter[i] = u_

with open('data/data_test/indomain.txt', 'w', encoding='utf-8') as file:
    for line in utter:
        file.write(line + '\n')