import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, bidirectional=True):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=bidirectional)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.enc_hid_dim = enc_hid_dim

    def forward(self, src, seq_len):
        # src : [src len, batch_size]
        # seq_len : [batch_size]
        src_id = src.transpose(0, -1)
        b_size = src_id.size(1)

        hidden = self.init_hidden(b_size=b_size, device=src_id.device)
        embedded = self.embedding(src_id)  # [seq_len, b_size, hidden_dim]
        output = F.relu(embedded)
        output = self.dropout(output)

        packed_emb = nn.utils.rnn.pack_padded_sequence(output, seq_len, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_emb, hidden)

        outputs, length = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = hidden.mean(dim=0)
        # output: [batch_size, seq_len, hidden_size]
        return outputs, hidden

    def init_hidden(self, b_size, device):
        weight = next(self.parameters()).data
        if self.bidirectional:
            return Variable(weight.new(2, b_size, self.enc_hid_dim).zero_()).to(device)
        else:
            return Variable(weight.new(1, b_size, self.enc_hid_dim).zero_()).to(device)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.5, embedding=None):
        super().__init__()

        self.output_dim = output_dim
        if not embedding:
            self.embedding = nn.Embedding(output_dim, emb_dim)
        else:
            self.embedding = embedding

        self.rnn = nn.GRU(enc_hid_dim, dec_hid_dim)
        self.fc_out = nn.Linear(enc_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, seq_len, hidden):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # tgt_id = tgt.transpose(0, -1)
        tgt_id = tgt
        hidden = hidden.unsqueeze(0)

        embedded = self.embedding(tgt_id)  # [seq_len, b_size, hidden_dim]
        output = F.relu(embedded)
        output = self.dropout(output)

        outputs, hidden = self.rnn(output.transpose(1, 0), hidden)
        outputs = outputs.transpose(1, 0)
        outputs = self.fc_out(outputs)
        # output: [batch_size, seq_len, hidden_size]
        return outputs


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.5):
        super().__init__()
        self.enc = Encoder(vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, dropout)
        self.dec = Decoder(vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, dropout, embedding=self.enc.embedding)

    def forward(self, src, src_len, tgt, tgt_len):
        _, hidden = self.enc(src, src_len)
        outp = self.dec(tgt, tgt_len, hidden)
        return outp
