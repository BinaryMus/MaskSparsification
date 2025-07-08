import torch
from scipy.stats import norm

class Baseline:
    def __init__(self, **kwargs):
        pass

    def encode(self, x):
        return x, 1
    
    def decode(self, x, y):
        return x.clone().detach()

class VanillaQuantization:
    def __init__(self, bit, **kwargs) -> None:
        self.bit = bit

    def encode(self, _x):
        with torch.no_grad():
            x = _x.detach().clone()
            min_value, max_value = x.min().item(), x.max().item()
            k = ((2 ** self.bit) - 1) / (max_value - min_value)
            b = -min_value * k
            y = torch.round(k * x + b)
        return y, k, b

    def decode(self, y, k, b):
        with torch.no_grad():
            return (y - b) / k

class QuantileQuantization:
    def __init__(self, bit, offset, asymmetric=False, stats='norm', **kwargs) -> None:
        self.bit = bit
        self.offset = offset
        if stats == 'norm':
            from scipy.stats import norm
            self.ppf = norm.ppf
        else:
            raise NotImplementedError
        self.quantiles = self.create_normal_map(bit, offset, asymmetric)
        
    def encode(self, _x):
        with torch.no_grad():
            x = _x.detach().clone()
            max_value = x.abs().max().item()
            sz = x.shape
            res = x / max_value
            self.quantiles = self.quantiles.to(_x.device)
            positions = torch.searchsorted(self.quantiles, res.view(-1))
            positions = torch.clamp(positions, 0, len(self.quantiles) - 1)

            left = self.quantiles[torch.clamp(positions - 1, 0)]
            right = self.quantiles[positions]

            closest_index = torch.where(
                (res.view(-1) - left).abs() < (res.view(-1) - right).abs(),
                positions - 1,
                positions
            )

            res = closest_index.view(sz)
        return res, max_value

    def decode(self, x, max_value):
        with torch.no_grad():
            return self.quantiles[x.long()] * max_value

    def create_normal_map(self, bit=4, offset=0.8, asymmetric=False):
        bins = 2 ** bit
        if asymmetric:
            v1 = self.ppf(torch.linspace(offset, 0.5, bins // 2)[:-1]).tolist()[::-1]
            v2 = [0, 0]
            v3 = (-self.ppf(torch.linspace(offset, 0.5, bins // 2)[:-1])).tolist()
            v = v3 + v2 + v1
        else:
            v1 = self.ppf(torch.linspace(offset, 0.5, bins // 2 + 1)[:-1]).tolist()[::-1]
            v2 = [0]
            v3 = (-self.ppf(torch.linspace(offset, 0.5, bins // 2)[:-1])).tolist()
            v = v3 + v2 + v1

        values = torch.Tensor(v)
        values /= values.max()
        return values
    
class FpQuantization:
    def __init__(self, exponent_bits=4, mantissa_bits=3, **kwargs):
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits
        self.bias = (2 ** (self.exponent_bits - 1)) - 1

    def encode(self, _x):
        with torch.no_grad():
            x = _x.detach().clone()
            sign = torch.sign(x)
            x = torch.abs(x).view(-1)
            
            exponent = torch.floor(torch.log2(x + 1e-8))
            mantissa = x / 2 ** exponent - 1
            exponent_q = torch.clamp(exponent + self.bias, 0, 2 ** self.exponent_bits - 1)
            mantissa_q = torch.round(mantissa * (2 ** self.mantissa_bits))
            return sign, exponent_q, mantissa_q

    def decode(self, sign, exponent_q, mantissa_q):
        with torch.no_grad():
            x = (2 ** (exponent_q - self.bias)) * (1 + mantissa_q / (2 ** self.mantissa_bits))
            x = x.view(sign.shape)
            return sign * x

class VanillaSparsification:
    def __init__(self, ratio, k=None, **kwargs) -> None:
        self.ratio = ratio
        self.k = k

    def encode(self, _x):
        with torch.no_grad():
            x = _x.detach().clone()
            sz = x.shape
            x = x.view(-1)
            if self.k is None:
                k = int(self.ratio * x.size(0))
            else:
                k = self.k
            values, indices = torch.topk(x, k=k)
        return sz, values, indices
    
    def decode(self, sz, values, indices):
        with torch.no_grad():
            x = torch.zeros(sz).view(-1).to(values.device)
            x[indices] = values
            x = x.view(sz)
        return x

class MaskSparsification:
    def __init__(self, ratio, bit, k=None, positive=False, **kwargs) -> None:
        self.ratio = ratio
        self.bit = bit
        self.k = k
        self.positive = positive
        if not positive:
            self.bit -= 1

    def encode(self, _x):
        with torch.no_grad():
            mask = _x.detach().clone()
            sz = mask.shape
            sign = None if self.positive else mask.sign()
            mask = mask.abs().view(-1)
            if self.k is None:
                k = int(self.ratio * mask.size(0))
            else:
                k = self.k
            values, indices = torch.topk(mask, k=k)
            k = ((1 << self.bit) - 1) / values[-1]
            mask = torch.round(k * mask)
            mask = torch.clamp(mask, 0, (1 << self.bit) - 2)
            mask[indices] = (1 << self.bit) - 1
            _, value_positions = torch.sort(indices)
            vector = values.index_select(0, value_positions)
        return sz, vector, mask, sign
    
    def decode(self, sz, vector, mask, sign):
        with torch.no_grad():
            idx = mask == (1 << self.bit) - 1
            mask *= (vector.min() / ((1 << self.bit) - 1))
            mask[idx] = vector
            mask = mask.view(sz)
            if not self.positive:
                mask *= sign
        return mask


class RandTopkSparsification:
    def __init__(self, ratio, k=None, alpha=0.1, **kwargs) -> None:
        self.ratio = ratio
        self.alpha = alpha
        self.k = k

    def encode(self, _x):
        with torch.no_grad():
            x = _x.detach().clone()
            sz = x.shape
            x = x.view(-1)
            if self.k is None:
                k = int(self.ratio * x.size(0))
            else:
                k = self.k
            _, top_k_indices = torch.topk(x, k)
            N1, N2 = k, x.size(0) - k
            probabilities = torch.ones_like(x)
            probabilities[top_k_indices] = (1 - self.alpha) / N1
            probabilities[torch.where(probabilities == 1)] = self.alpha / N2
            selected_neurons = torch.multinomial(probabilities, k, replacement=False)
            return sz, x[selected_neurons], selected_neurons
        
    def decode(self, sz, values, indices):
        with torch.no_grad():
            x = torch.zeros(sz).view(-1).to(values.device)
            x[indices] = values
            x = x.view(sz)
        return x
