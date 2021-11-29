from en.model_types import PersonName
from en.model import MyNet, load_model_en
from model_types import PersonCountryOfBirth
import string
from torch.nn.utils.rnn import pad_sequence
import torch
from contextlib import contextmanager


class NameData(torch.utils.data.Dataset):
    def __init__(self, names, allowed_chars):
        self.names = names
        self.vocab = {c: it for it, c in enumerate(allowed_chars)}
        self.num_chars = len(self.vocab)

    def encode_name(self, name):
        return torch.tensor([self.vocab[c] for c in name])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.encode_name(self.names[index])
        return name


def pad_batch(batch):
    x_lens = [len(x) for x in batch]
    x_pad = pad_sequence(batch, batch_first=True, padding_value=0)
    return x_pad, x_lens


global country_classes_rev
country_classes_rev = None


def get_class_name(x):
    global country_classes_rev
    if not country_classes_rev:
        _, country_classes_rev = load_model_en()
    return country_classes_rev.get(x, None)


@contextmanager
def infer(name: PersonName, model: MyNet) -> PersonCountryOfBirth:
    global country_classes_rev
    vocab = string.ascii_lowercase + " -"
    dataset = NameData([name], vocab)

    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=pad_batch, batch_size=1)

    for (x, x_len) in data_loader:
        try:
            p = torch.argmax(model(x, x_len), axis=-1)
            p = list(map(get_class_name, p.detach().numpy()))
            yield PersonCountryOfBirth(first_name=name, country_of_birth=p[0])
        finally:
            model.eval()
