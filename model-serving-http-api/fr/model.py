from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch
from en.model import MyNet


class MyNetFR(MyNet):
    def __init__(self, num_classes, vocab_size, hidden_size, embed_size):
        super(MyNet, self).__init__()
        self.dim = hidden_size
        self.embed_size = embed_size

        self.fr_embedding = nn.Embedding(vocab_size, self.embed_size)
        self.lstm = nn.RNN(input_size=self.embed_size, hidden_size=self.dim, num_layers=1, batch_first=True)

        self.fc = nn.Linear(self.dim, num_classes)

    def forward(self, x, xlen):

        x = pad_sequence(x, batch_first=True)
        x = self.fr_embedding(x)
        x = pack_padded_sequence(x, xlen, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[..., -1, :]
        x = F.relu(x)
        x = self.fc(x)
        return x


global model_fr

model_fr = None


def load_model_fr():
    model = torch.load("./.models/rnn-fr.pt")
    model = model["model"]
    global model_fr
    model_fr = model
    return model
