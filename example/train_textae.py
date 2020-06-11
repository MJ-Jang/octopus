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


from octopus.classifier import TextCNNAE

aa = TextCNNAE('basic_tokenizer.model', 128, 128)
train_config = {
    'sents': utterances,
    'batch_size': 256,
    'num_epochs': 5,
    'lr': 0.001,
    'save_path': './',
    'model_prefix': 'test',
    'max_len': 10,
    'num_workers': 4
}

aa.train(**train_config)
aa.save_dict('example/test.model')
