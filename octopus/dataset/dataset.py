import torch
import numpy as np

from torch.utils.data.dataset import Dataset

from typing import Text, Dict, List

PAD_TOKEN = '<pad>'
START_TOKEN = '<s>'
END_TOKEN = '</s>'


class EncoderDecoderDataset(Dataset):

    name = 'encoder_decoder_dataset'

    def __init__(self,
                 tok,
                 inputs: List,
                 targets: List,
                 max_len: int,
                 pad_token: Text = PAD_TOKEN,
                 start_token: Text = START_TOKEN,
                 end_token: Text = END_TOKEN):

        assert len(inputs) == len(targets)

        self.inputs = inputs
        self.targets = targets
        self.tok = tok
        self.max_len = max_len

        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sent = self.inputs[idx]
        seq_len = min(len(self.tok.encode_as_ids(input_sent)), self.max_len)
        target_sent = self.targets[idx]

        input_token = np.array(self.tokenize(input_sent))
        target_token = np.array(self.tokenize(target_sent))

        return input_token, target_token, seq_len

    def tokenize(self, text: Text):
        pad_id = self.tok.piece_to_id(self.pad_token)
        start_id = self.tok.piece_to_id(self.start_token)
        end_id = self.tok.piece_to_id(self.end_token)

        token = self.tok.encode_as_ids(text)
        if len(token) < self.max_len - 2:
            token += [end_id] + [pad_id] * (self.max_len - len(token) - 2)
            token = [start_id] + token
        else:
            token = [start_id] + token[:self.max_len-2] + [end_id]
        return token


class AEDataset(Dataset):
    def __init__(self, tokenizer, sents: list, max_len: int):
        self.tok = tokenizer
        self.data = sents
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sent = self.data[item]
        tokens = self.tok.text_to_id(sent)
        if len(tokens) < self.max_len:
            tokens += [self.tok.pad] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        return torch.LongTensor(tokens), torch.LongTensor(tokens)