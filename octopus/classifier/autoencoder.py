
from octopus.tokenizer import SoyTokenizer
from octopus.module import TextCNN
from octopus.dataset import AEDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
from difflib import SequenceMatcher

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dill
import os
import re


class TextCNNAE:
    def __init__(self,
                 tokenizer_path: str,
                 embedding_dim: int,
                 hidden_dim: int,
                 use_gpu: bool = True, **kwargs):
        self.device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'
        if self.device == 'cuda:0':
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0

        self.tok = SoyTokenizer(tokenizer_path)
        self.vocab_size = len(self.tok)

        self.model_conf = {
            'vocab_size': self.vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'n_class': self.vocab_size,
        }

        self.model = TextCNN(**self.model_conf)

        if self.n_gpu == 1:
            self.model = self.model.cuda()
        elif self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()

    def train(self,
              sents: list,
              batch_size: int,
              num_epochs: int,
              lr: float,
              save_path: str = None,
              model_prefix: str = None,
              max_len: int = 8,
              num_workers: int = 4
              ):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        dataset = AEDataset(self.tok, sents, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        for epoch in range(num_epochs):
            total_loss = 0
            for context, target in tqdm(dataloader, desc='batch progress'):
                # Remember PyTorch accumulates gradients; zero them out
                self.model.zero_grad()

                context = context.to(self.device)
                target = target.to(self.device)

                logits = self.model(context)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.reshape(-1),
                                       ignore_index=self.tok.pad)

                # backpropagation
                loss.backward()
                # update the parameters
                optimizer.step()
                total_loss += loss.item()
            print("Total loss: {}".format(round(total_loss, 3)))

    def infer(self, text: str):
        text = self._preprocess([text])[0]
        inputs = self.tok.text_to_id(text)
        inputs = torch.LongTensor([inputs])

        logits = self.model(inputs)
        pred = logits.argmax(dim=-1)
        outp = self.tok.id_to_text(pred.tolist()[0])

        ratio = SequenceMatcher(None, text.replace(' ', ''), ''.join(outp)).ratio()
        print(ratio)
        print(outp)

    def save_dict(self, save_path: str, model_prefix: str):
        os.makedirs(save_path, exist_ok=True)

        filename = os.path.join(save_path, model_prefix+'.modeldict')

        outp_dict = {
            'model_params': self.model.cpu().state_dict(),
            'model_conf': self.model_conf,
            'model_type': 'pytorch'
        }

        with open(filename, "wb") as file:
            dill.dump(outp_dict, file, protocol=dill.HIGHEST_PROTOCOL)
        self.model.to(self.device)

    def load_model(self, model_path: str):
        with open(model_path, 'rb') as modelFile:
            model_dict = dill.load(modelFile)
        model_conf = model_dict['model_conf']
        self.model = TextCNN(**model_conf)
        try:
            self.model.load_state_dict(model_dict["model_params"])
        except:
            new_dict = OrderedDict()
            for key in model_dict["model_params"].keys():
                new_dict[key.replace('module.', '')] = model_dict["model_params"][key]
            self.model.load_state_dict(new_dict)

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _preprocess(sents: list):
        n_str_pattern = re.compile(pattern='[\\d\\-?/_!\\.,]')
        doublespacing = re.compile(pattern='\\s\\s+')

        sents = [n_str_pattern.sub(repl=' ', string=w) for w in sents]
        sents = [doublespacing.sub(repl=' ', string=w).strip() for w in sents]
        sents = [u.lower() for u in sents]
        return sents
