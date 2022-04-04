import torch.nn as nn
import torch


class MyNet(nn.Module):
    """
    Mock English model
    """

    def __init__(self, num_classes, vocab_size, hidden_size, embed_size):
        super(MyNet, self).__init__()
        pass

    def forward(self, x, xlen):
        return x


global model_en

model_en = None


def load_model_en():
    model = torch.load("./.models/country-prediction-en.pt")
    country_classes = model["classes"]
    model = model["model"]
    global model_en, country_classes_rev
    model_en = model
    country_classes_rev = dict([(v, k) for k, v in country_classes.items()])
    return model, country_classes_rev
