from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch


class MyNet(nn.Module):
    def __init__(self, num_classes, vocab_size, hidden_size, embed_size):
        super(MyNet, self).__init__()
        self.dim = hidden_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.lstm = nn.RNN(input_size=self.embed_size, hidden_size=self.dim, num_layers=1, batch_first=True)

        self.fc = nn.Linear(self.dim, num_classes)

    def forward(self, x, xlen):

        x = pad_sequence(x, batch_first=True)
        x = self.embedding(x)
        x = pack_padded_sequence(x, xlen, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[..., -1, :]
        x = F.relu(x)
        x = self.fc(x)
        return x


global model_en

model_en = None


def load_model_en():
    model = torch.load("./.models/rnn-en.pt")
    country_classes = model["classes"]
    model = model["model"]
    global model_en, country_classes_rev
    model_en = model
    country_classes_rev = dict([(v, k) for k, v in country_classes.items()])
    return model, country_classes_rev
