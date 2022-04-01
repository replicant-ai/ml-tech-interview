import torch
from en.model import MyNet


class MyNetFR(MyNet):
    """
    Mock French model
    """

    def __init__(self, num_classes, vocab_size, hidden_size, embed_size):
        super(MyNet, self).__init__()
        pass

    def forward(self, x, xlen):
        return x


global model_fr

model_fr = None


def load_model_fr():
    model = torch.load("./.models/country-prediction-fr.pt")
    model = model["model"]
    global model_fr
    model_fr = model
    return model
