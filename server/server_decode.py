import torch


class DecodeQuantizer:
    def __init__(self, bit: int = 8, **kwargs):
        self.bit = bit

    def __str__(self):
        return f"{self.bit}-bit quantization"

    def decode(self, x: torch.Tensor, scale: (float, float), data):
        with torch.no_grad():
            if data.device != x.device:
                data = data.to(x.device)
            data = data.to(torch.float32)
            k, b = scale
            data -= b
            data /= k
            x.data = data


class DecodeSparser:
    def __init__(self, ratio: float = 0.75, **kwargs):
        self.ratio = ratio

    def __str__(self):
        return f"Top-{int((1 - self.ratio) * 100)}% sparsification"

    def decode(self, x: torch.Tensor, values: torch.Tensor, indices: torch.Tensor) -> None:
        with torch.no_grad():
            if values.device != x.device:
                values = values.to(x.device)
                indices = indices.to(x.device)
            vector = torch.zeros_like(x).view(-1)
            vector[indices] = values
            x.data = vector.view(x.size())


class DecodeMaskedSparser:
    def __init__(self, ratio: float = 0.98, bit: int = 2):
        self.ratio = ratio
        self.bit = bit
        self.sparser = DecodeSparser(ratio)
        self.quantizer = DecodeQuantizer(bit)
        self.length = (1 << bit) - 1

    def __str__(self):
        return f"Top-{int((1 - self.ratio) * 100)}% sparsification" \
               f"with ({self.bit}-bit) mask"

    def decode(self, x: torch.Tensor, vector: torch.Tensor, mask: torch.Tensor):
        with torch.no_grad():
            if vector.device != x.device:
                vector = vector.to(x.device)
                mask = mask.to(x.device)
            mask = mask.to(torch.float32)
            idx = mask == self.length
            mask *= (vector.min() / self.length)
            mask[idx] = vector
            x.data = mask.view(x.size())
