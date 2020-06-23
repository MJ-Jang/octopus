from octopus.tokenizer import SentencePieceTokenizer
from octopus.dataset import EncoderDecoderDataset
from octopus.module.gru import Seq2Seq, Encoder, Decoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dill
import os
import re


class Seq2SeqAE:
    def __init__(self,
                 tokenizer_path: str,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float = 0.5,
                 use_gpu: bool = True, **kwargs):

        self.device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'

        self.tok = SentencePieceTokenizer(tokenizer_path)
        self.vocab_size = len(self.tok)
        self.tok_name = tokenizer_path.split('/')[-1]
        self.max_len = None

        self.model_conf = {
            'vocab_size': self.vocab_size,
            'emb_dim': enc_hid_dim,
            'enc_hid_dim': enc_hid_dim,
            'dec_hid_dim': dec_hid_dim,
            "dropout": dropout
        }
        self.model = Seq2Seq(**self.model_conf)
        if self.device == 'cuda:0':
            self.n_gpu = torch.cuda.device_count()
            self.model.cuda()
        else:
            self.n_gpu = 0

    def train(self,
              sents: list,
              batch_size: int,
              num_epochs: int,
              lr: float,
              max_len: int = 8,
              num_workers: int = 4
              ):
        self.model.train()
        self.max_len = max_len
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        dataset = EncoderDecoderDataset(tok=self.tok, inputs=sents, targets=sents, max_len=max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc='batch progress'):
                # Remember PyTorch accumulates gradients; zero them out
                inputs, input_len, target_inputs, target_outputs = batch
                self.model.zero_grad()

                inputs = inputs.to(self.device)
                input_len = input_len.to(self.device)
                target_inputs = target_inputs.to(self.device)
                target_outputs = target_outputs.to(self.device)
                logits = self.model(inputs, input_len, target_inputs, input_len)

                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_outputs.reshape(-1),
                                       ignore_index=self.tok.token_to_id(self.tok.pad))

                # backpropagation
                loss.backward()
                # update the parameters
                optimizer.step()
                total_loss += loss.item()
            print("Total loss: {}".format(round(total_loss, 3)))

    def infer(self, text: str):
        pass

    def save_dict(self, save_path: str, model_prefix: str):
        os.makedirs(save_path, exist_ok=True)

        filename = os.path.join(save_path, model_prefix+'.modeldict')

        outp_dict = {
            'tok_name': self.tok_name,
            'max_len': self.max_len,
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
        self.model = Seq2Seq(**model_conf)
        try:
            self.model.load_state_dict(model_dict["model_params"])
        except:
            new_dict = OrderedDict()
            for key in model_dict["model_params"].keys():
                new_dict[key.replace('module.', '')] = model_dict["model_params"][key]
            self.model.load_state_dict(new_dict)

        self.max_len = model_dict.get('max_len')
        self.model.to(self.device)
        self.model.eval()

    def _generate_vectors(self, sents: list, batch_size: int = 128, do_average: bool = True):
        dataset = EncoderDecoderDataset(tok=self.tok, inputs=sents, targets=sents, max_len=8)
        loader = DataLoader(dataset, batch_size=batch_size)
        outp = []
        for i, b in enumerate(loader):
            inputs, input_len, _, _ = b
            _, vecs = self.model.enc(inputs, input_len)
            outp.append(vecs)
        outp = torch.cat(outp, dim=0)
        if do_average:
            outp = outp.mean(dim=0)
        outp = outp.detach().cpu().numpy()
        return outp

    def _generate_single_vector(self, sent: str):
        dataset = EncoderDecoderDataset(tok=self.tok, inputs=[sent], targets=[sent], max_len=8)
        inputs, input_len, _, _ = dataset.__getitem__(0)
        _, vecs = self.model.enc(torch.LongTensor([inputs]), [input_len])
        vecs = vecs.detach().cpu().numpy()
        return vecs

    def construct_centervecs(self, utter_dict: dict):
        """
        utter_dict: {
            "intent_name": [
                "aaa",
                "bbb"
            ]
        }
        """
        outp = []
        for key, sents in tqdm(utter_dict.items()):
            center_vec = self._generate_vectors(sents, batch_size=128)
            outp.append(center_vec)
        return outp