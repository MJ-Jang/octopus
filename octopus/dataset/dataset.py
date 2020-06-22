import torch
import numpy as np

from torch.utils.data.dataset import Dataset

from typing import Text, Dict, List

UNK_TOKEN = '<unk>'
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

        self.unk_id = tok.token_to_id(UNK_TOKEN)
        self.pad_id = tok.token_to_id(PAD_TOKEN)
        self.sos_id = tok.token_to_id(START_TOKEN)
        self.eos_id = tok.token_to_id(END_TOKEN)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sent = self.inputs[idx]
        input_len = min(len(self.tok.text_to_id(input_sent)), self.max_len)
        target_sent = self.targets[idx]

        input_token = np.array(self.tokenize(input_sent))
        target_token = self.tokenize(target_sent)

        target_inputs = np.array(target_token[:-1] + [self.pad_id])
        target_outputs = np.array(target_token[1:] + [self.pad_id])

        return input_token, input_len, target_inputs, target_outputs

    def tokenize(self, text: Text):
        token = self.tok.text_to_id(text)
        if len(token) < self.max_len - 2:
            token += [self.eos_id] + [self.pad_id] * (self.max_len - len(token) - 2)
            token = [self.sos_id] + token
        else:
            token = [self.sos_id] + token[:self.max_len - 2] + [self.eos_id]
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