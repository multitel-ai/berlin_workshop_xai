import torch
from torch.nn import MultiheadAttention
from types import MethodType

from utils import rsetattr


class WrappedMultiheadAttention(torch.nn.Module):
    def __init__(self, multihead_attention, name, attention_forward_hook):
        super(WrappedMultiheadAttention, self).__init__()
        self.attn = multihead_attention
        self.name = name
        self.attention_forward_hook = attention_forward_hook

    def forward(self, query, key, value, **kwargs):
        if "need_weight" not in kwargs:
            need_weight = False
        else:
            need_weight = kwargs["need_weight"]
        kwargs["need_weights"] = True
        x, attention_weight_output = self.attn(query, key, value, **kwargs)
        self.attention_forward_hook(self.name, attention_weight_output)
        if need_weight:
            return x, attention_weight_output
        else:
            return x, None


def wrap_transformer(model):
    model.attention_weights = {}

    def set_attention_weights(self, layer_name, attention_weights):
        self.attention_weights[layer_name] = attention_weights

    model.set_attention_weights = MethodType(set_attention_weights, model)

    for name, module in list(model.named_modules()):
        if isinstance(module, MultiheadAttention):
            model.attention_weights[name] = None
            new_module = WrappedMultiheadAttention(module, name, model.set_attention_weights)
            rsetattr(model, name, new_module)