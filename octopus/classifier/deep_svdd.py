import os
import dill
import torch.optim as optim
import torch
import numpy as np

from octopus.tokenizer import SentencePieceTokenizer
from octopus.dataset import DeepSVDDDataset
from collections import OrderedDict
from octopus.module.gru import Encoder
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


class TextDeepSVDD:
    def __init__(self,
                 tokenizer_path: str,
                 enc_hid_dim: int,
                 dropout: float = 0.5,
                 use_gpu: bool = True, **kwargs):

        self.device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'

        self.tok = SentencePieceTokenizer(tokenizer_path)
        self.vocab_size = len(self.tok)
        self.tok_name = tokenizer_path.split('/')[-1]
        self.max_len = None

        self.model_conf = {
            'input_dim': self.vocab_size,
            'emb_dim': enc_hid_dim,
            'enc_hid_dim': enc_hid_dim,
            'dec_hid_dim': enc_hid_dim,
            "dropout": dropout
        }
        self.model = Encoder(**self.model_conf)

        if self.device == 'cuda:0':
            self.n_gpu = torch.cuda.device_count()
            self.model.cuda()
            self.c = Variable(torch.Tensor(self._xavier_init(enc_hid_dim)).cuda(), requires_grad=True)
        else:
            self.n_gpu = 0
            self.c = Variable(torch.Tensor(self._xavier_init(enc_hid_dim)), requires_grad=True)

    def train(self,
              sents: list,
              batch_size: int,
              num_epochs: int,
              lr: float,
              max_len: int = 8,
              lamb: float = 1e-5,
              num_workers: int = 4
              ):

        self.model.train()
        self.max_len = max_len
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        dataset = DeepSVDDDataset(tokenizer=self.tok, sents=sents, max_len=max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc='batch progress'):
                # Remember PyTorch accumulates gradients; zero them out
                inputs, input_len = batch
                self.model.zero_grad()

                inputs = inputs.to(self.device)
                input_len = input_len.to(self.device)

                _, vecs = self.model(inputs, input_len)

                l_c = 1 / len(vecs) * torch.sum((vecs - self.c) ** 2)
                frob_reg = torch.tensor(0.).to(self.device)
                for param in self.model.parameters():
                    frob_reg += torch.norm(param, p='fro').to(self.device)
                loss = l_c + lamb / 2 * frob_reg

                # backpropagation
                loss.backward()
                # update the parameters
                optimizer.step()
                total_loss += loss.item()
            print("Total loss: {}".format(round(total_loss, 3)))

    def save_model(self, save_path, model_prefix):
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, model_prefix+'.modeldict')

        outp_dict = {
            'max_len': self.max_len,
            'model_params': self.model.cpu().state_dict(),
            'model_conf': self.model_conf,
            'model_type': 'pytorch',
            'centroid': self.c
        }

        with open(filename, "wb") as file:
            dill.dump(outp_dict, file, protocol=dill.HIGHEST_PROTOCOL)
        self.model.to(self.device)

    def load_model(self, model_path: str):
        with open(model_path, 'rb') as modelFile:
            model_dict = dill.load(modelFile)
        model_conf = model_dict['model_conf']
        self.model = Encoder(**model_conf)
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
        self.c = model_dict.get('centroid')

    @staticmethod
    def _xavier_init(n_in: int, n_out: int = None):
        if not n_out:
            n_out = n_in
        return np.random.uniform(-np.sqrt(6/(n_in + n_out)), np.sqrt(6/(n_in + n_out)), [n_in])
