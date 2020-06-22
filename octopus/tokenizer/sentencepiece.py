# Improved soynlp LTokenizer for this package
import sentencepiece as sp

from sentencepiece import SentencePieceProcessor, SentencePieceTrainer


class SentencePieceTokenizer:
    def __init__(self, model_path: str = None):
        self.unk = '<unk>'
        self.pad = '<pad>'
        self.sos = '<s>'
        self.eos = '</s>'

        if model_path:
            self.load(model_path)
        else:
            self.tokenizer = None

    def tokenize(self, sent: str):
        return self.tokenizer.encode_as_pieces(sent)

    def text_to_id(self, sent: str):
        return self.tokenizer.encode_as_ids(sent)

    def id_to_text(self, idxs: list):
        return self.tokenizer.decode_ids(idxs)

    def token_to_id(self, token: str):
        return self.tokenizer.piece_to_id(token)

    def train(self,
              sent_path: str,
              model_prefix: str,
              character_coverage=0.9995,
              vocab_size=None,
              model_type: str = "bpe",
              control_symbols: list = ['<pad>']
              ):

        if character_coverage is None and vocab_size is None:
            print("at least character_coverage or vocab_size should be given!")
            assert character_coverage or vocab_size

        coverage_conditions = ""
        if character_coverage is not None:
            coverage_condition = f" --character_coverage={str(character_coverage)} "
        else:
            coverage_condition = f" --vocab_size={str(vocab_size)} "

        symbol_list = ""
        for i in control_symbols:
            symbol_list += i + ","

        args = (
            "--input={} "
            "--model_prefix={} "
            "--model_type={} "
            "--control_symbols={} ".format(
                sent_path, model_prefix, model_type, symbol_list
            )
        )

        args += coverage_condition

        SentencePieceTrainer.Train(args)

    def load(self, model_path: str):
        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.Load(model_path)

    def __repr__(self):
        unk = '"{}"'.format(self.unk) if self.unk else "None"
        return "Vocab(size={}, unk={}, pad={})".format(
            len(self.tokenizer), unk, self.pad)

    def __len__(self):
        return len(self.tokenizer)
