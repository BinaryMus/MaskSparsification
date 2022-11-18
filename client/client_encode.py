import torch


class EncodeQuantizer:
    def __init__(self, bit: int = 8, **kwargs):
        self.bit = bit

    def __str__(self):
        return f"{self.bit}-bit quantization"

    def encode(self, x: torch.Tensor, mi=None, ma=None):
        with torch.no_grad():
            if ma is None:
                ma = x.data.max().item()
            if mi is None:
                mi = x.data.min().item()
            k = ((1 << self.bit) - 1) / (ma - mi)
            b = -mi * k
            data = torch.round(k * x.data + b).to(torch.uint8)
        return (k, b), data


class EncodeSparser:
    def __init__(self, ratio: float = 0.75, **kwargs):
        self.ratio = ratio

    def __str__(self):
        return f"Top-{int((1 - self.ratio) * 100)}% sparsification"

    def encode(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            data_vec = x.data.view(-1)
            values, indices = torch.topk(data_vec, k=int(data_vec.size(0) * (1 - self.ratio)))
        return values, indices


class EncodeSparserWithCompensation:
    def __init__(self, ratio: float = 0.98, bit: int = 2):
        self.ratio = ratio
        self.bit = bit
        self.sparser = EncodeSparser(ratio)
        self.quantizer = EncodeQuantizer(bit)
        self.length = (1 << bit) - 1

    def __str__(self):
        return f"Top-{int((1 - self.ratio) * 100)}% sparsification" \
               f"with ({self.bit}-bit) mask immediately compensation"

    def encode(self, x: torch.Tensor):
        with torch.no_grad():
            mask = x.data.view(-1)
            values, indices = self.sparser.encode(x)
            _, mask = self.quantizer.encode(mask, 0, values[-1])
            mask[mask == self.length] = self.length - 1
            mask[indices] = self.length
            _, value_positions = torch.sort(indices)
            vector = values.index_select(0, value_positions)
            return vector, mask
