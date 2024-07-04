import torch


class EncodeQuantizer:
    def __init__(self, bit: int = 8, **kwargs):
        self.bit = bit

    def __str__(self):
        return f"{self.bit}-bit quantization"

    def __call__(self, x: torch.Tensor, mi=None, ma=None):
        with torch.no_grad():
            if ma is None:
                ma = x.data.max().item()
            if mi is None:
                mi = x.data.min().item()
            k = ((1 << self.bit) - 1) / (ma - mi)
            b = -mi * k
            data = torch.round(k * x.data + b)
        return (k, b), data


class EncodeSparser:
    def __init__(self, k: int, **kwargs):
        self.k = k

    def __str__(self):
        return f"Top-{self.k} sparsification"

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            data_vec = x.data.view(-1)
            values, indices = torch.topk(data_vec, k=self.k)
        return values, indices


class EncodeMaskedSparser:
    def __init__(self, k: int, bit: int = 2):
        self.k = k
        self.bit = bit
        self.sparser = EncodeSparser(k)
        self.quantizer = EncodeQuantizer(bit)
        self.length = (1 << bit) - 1

    def __str__(self):
        return f"Top-{self.k} sparsification with {self.bit}-bit mask"

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            mask = x.data.view(-1)
            values, indices = self.sparser(x)
            _, mask = self.quantizer(mask, 0, values[-1])
            mask[mask == self.length] = self.length - 1
            mask[indices] = self.length
            _, value_positions = torch.sort(indices)
            vector = values.index_select(0, value_positions)
            return vector, mask
